"""
Training script for Jewelry Retrieval Model
Optimized for maximum performance with OpenCLIP ViT-L/14
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime

from config import Config
from dataset import get_dataloader
from model import build_model
from losses import get_loss_function, CombinedLoss
from utils import set_seed, save_checkpoint, load_checkpoint, AverageMeter


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, config, writer, use_ce_only=False, num_epochs=100):
    """Train for one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    supcon_loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if config.MIXED_PRECISION:
            with autocast('cuda'):
                # Forward pass
                if isinstance(criterion, CombinedLoss) or use_ce_only:
                    # Need logits for combined loss or CE-only
                    if isinstance(model, nn.DataParallel):
                        embeddings, logits = model.module(images, return_embeddings=True)
                    else:
                        embeddings, logits = model(images, return_embeddings=True)
                    
                    if use_ce_only:
                        # CE-only mode
                        loss = nn.CrossEntropyLoss()(logits, labels)
                        loss_dict = {'ce': loss.item(), 'supcon': 0.0}
                    else:
                        # Combined loss
                        loss, loss_dict = criterion(embeddings, logits, labels)
                else:
                    # SupCon or Triplet only
                    if isinstance(model, nn.DataParallel):
                        embeddings = model.module.get_embeddings(images)
                    else:
                        embeddings = model.get_embeddings(images)
                    loss = criterion(embeddings, labels)
                    loss_dict = {'ce': 0.0, 'supcon': loss.item()}
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            # Check gradients (once per epoch)
            if batch_idx == 0:
                grad_norm = check_gradients(model, epoch)
                writer.add_scalar('Train/GradNorm', grad_norm, epoch)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training without mixed precision
            if isinstance(criterion, CombinedLoss) or use_ce_only:
                if isinstance(model, nn.DataParallel):
                    embeddings, logits = model.module(images, return_embeddings=True)
                else:
                    embeddings, logits = model(images, return_embeddings=True)
                
                if use_ce_only:
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    loss_dict = {'ce': loss.item(), 'supcon': 0.0}
                else:
                    loss, loss_dict = criterion(embeddings, logits, labels)
            else:
                if isinstance(model, nn.DataParallel):
                    embeddings = model.module.get_embeddings(images)
                else:
                    embeddings = model.get_embeddings(images)
                loss = criterion(embeddings, labels)
                loss_dict = {'ce': 0.0, 'supcon': loss.item()}
            
            loss.backward()
            
            # Check gradients (once per epoch)
            if batch_idx == 0:
                grad_norm = check_gradients(model, epoch)
                writer.add_scalar('Train/GradNorm', grad_norm, epoch)
            
            optimizer.step()
        
        # Compute accuracy
        if isinstance(criterion, CombinedLoss) or use_ce_only:
            if isinstance(model, nn.DataParallel):
                _, logits = model.module(images, return_embeddings=True)
            else:
                _, logits = model(images, return_embeddings=True)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Update metrics
        loss_meter.update(loss.item(), images.size(0))
        if 'ce' in loss_dict:
            ce_loss_meter.update(loss_dict['ce'], images.size(0))
        if 'supcon' in loss_dict:
            supcon_loss_meter.update(loss_dict['supcon'], images.size(0))
        
        # Update progress bar
        acc = 100. * correct / total if total > 0 else 0.0
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'ce': f'{ce_loss_meter.avg:.4f}',
            'supcon': f'{supcon_loss_meter.avg:.4f}',
            'acc': f'{acc:.2f}%'
        })
        
        # Log to tensorboard
        if batch_idx % config.LOG_INTERVAL == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/Loss_Avg', loss_meter.avg, global_step)
            if 'ce' in loss_dict:
                writer.add_scalar('Train/CE_Loss', loss_dict['ce'], global_step)
            if 'supcon' in loss_dict:
                writer.add_scalar('Train/SupCon_Loss', loss_dict['supcon'], global_step)
            writer.add_scalar('Train/Accuracy', acc, global_step)
    
    train_acc = 100. * correct / total if total > 0 else 0.0
    return loss_meter.avg, ce_loss_meter.avg, supcon_loss_meter.avg, train_acc


def check_gradients(model, epoch):
    """Check gradient norms for debugging"""
    total_norm = 0.0
    param_count = 0
    
    if isinstance(model, nn.DataParallel):
        model_to_check = model.module
    else:
        model_to_check = model
    
    # Check classifier head gradients
    if hasattr(model_to_check, 'classifier'):
        for param in model_to_check.classifier.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    
    if epoch == 1 and param_count > 0:
        print(f"Classifier head grad norm: {total_norm:.6f} ({param_count} params)")
    
    return total_norm


def validate(model, dataloader, criterion, device, epoch, config, writer, use_ce_only=False):
    """Validate the model"""
    model.eval()
    
    loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    supcon_loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            if isinstance(criterion, CombinedLoss) or use_ce_only:
                if isinstance(model, nn.DataParallel):
                    embeddings, logits = model.module(images, return_embeddings=True)
                else:
                    embeddings, logits = model(images, return_embeddings=True)
                
                if use_ce_only:
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    loss_dict = {'ce': loss.item(), 'supcon': 0.0}
                else:
                    loss, loss_dict = criterion(embeddings, logits, labels)
                
                # Compute accuracy
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            else:
                if isinstance(model, nn.DataParallel):
                    embeddings = model.module.get_embeddings(images)
                else:
                    embeddings = model.get_embeddings(images)
                loss = criterion(embeddings, labels)
                loss_dict = {'ce': 0.0, 'supcon': loss.item()}
            
            loss_meter.update(loss.item(), images.size(0))
            if 'ce' in loss_dict:
                ce_loss_meter.update(loss_dict['ce'], images.size(0))
            if 'supcon' in loss_dict:
                supcon_loss_meter.update(loss_dict['supcon'], images.size(0))
    
    val_acc = 100. * correct / total if total > 0 else 0.0
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/CE_Loss', ce_loss_meter.avg, epoch)
    writer.add_scalar('Val/SupCon_Loss', supcon_loss_meter.avg, epoch)
    writer.add_scalar('Val/Accuracy', val_acc, epoch)
    
    print(f"Validation Loss: {loss_meter.avg:.4f} | CE: {ce_loss_meter.avg:.4f} | SupCon: {supcon_loss_meter.avg:.4f} | Acc: {val_acc:.2f}%")
    
    return loss_meter.avg


def get_optimizer(model, config):
    """Get optimizer"""
    if config.OPTIMIZER == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    
    return optimizer


def get_scheduler(optimizer, config, num_training_steps):
    """Get learning rate scheduler"""
    if config.SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=1e-6
        )
    elif config.SCHEDULER == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    else:
        scheduler = None
    
    return scheduler


def main():
    """Main training function"""
    # Initialize config
    config = Config()
    
    # Set random seed for reproducibility
    set_seed(config.SEED)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # TensorBoard writer
    log_dir = os.path.join(config.RESULTS_DIR, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, train_dataset = get_dataloader(
        config.TRAIN_DIR,
        config.INPUT_SIZE,
        config.BATCH_SIZE,
        is_train=True,
        use_augmentation=config.USE_AUGMENTATION,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader, val_dataset = get_dataloader(
        config.VAL_DIR,
        config.INPUT_SIZE,
        config.BATCH_SIZE,
        is_train=False,
        use_augmentation=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    num_classes = len(train_dataset.categories)
    print(f"Number of categories: {num_classes}")
    print(f"Categories: {train_dataset.categories}")
    
    # Automatic multi-phase training configuration
    AUTO_PHASE_TRAINING = True  # Enable automatic phase transitions
    TOTAL_EPOCH_BUDGET = config.NUM_EPOCHS
    PHASE1_MAX_EPOCHS = 10  # Max epochs for frozen backbone phase
    PHASE1_TARGET_ACC = 75.0  # Target accuracy to move to Phase 2
    PHASE2_MAX_EPOCHS = 30  # Max epochs for unfrozen backbone phase
    PHASE2_TARGET_ACC = 85.0  # Target accuracy to move to Phase 3 (SupCon)
    # Phase 3 uses the remaining global budget so total training is 100 epochs.
    
    # Phase tracking
    current_phase = 1
    phase1_complete = False
    phase2_complete = False
    
    # Build model - start with frozen backbone (Phase 1)
    print("\nBuilding model...")
    use_classifier = True
    model = build_model(config, num_classes=num_classes, use_classifier=use_classifier, freeze_backbone=True)
    model = model.to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Initialize optimizer for Phase 1 (frozen backbone)
    def create_optimizer_for_phase(phase, model, config):
        """Create optimizer based on current phase"""
        if phase == 1:
            # Phase 1: Only train classifier head
            if isinstance(model, nn.DataParallel):
                trainable_params = list(model.module.classifier.parameters())
            else:
                trainable_params = list(model.classifier.parameters())
            print(f"\n[Phase {phase}] Training only classifier head ({sum(p.numel() for p in trainable_params):,} parameters)")
            if config.OPTIMIZER == "adamw":
                return optim.AdamW(trainable_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
            else:
                return optim.SGD(trainable_params, lr=config.LEARNING_RATE, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
        elif phase == 2:
            # Phase 2: Unfreeze backbone with different LRs
            if isinstance(model, nn.DataParallel):
                backbone_params = list(model.module.base_model.visual_encoder.parameters())
                head_params = list(model.module.classifier.parameters())
            else:
                backbone_params = list(model.base_model.visual_encoder.parameters())
                head_params = list(model.classifier.parameters())
            
            backbone_lr = 1e-6  # Very low LR for pretrained backbone
            head_lr = 1e-4  # Higher LR for classifier head
            
            print(f"\n[Phase {phase}] Unfreezing backbone!")
            print(f"  Backbone ({sum(p.numel() for p in backbone_params):,} params) LR: {backbone_lr}")
            print(f"  Head ({sum(p.numel() for p in head_params):,} params) LR: {head_lr}")
            
            if config.OPTIMIZER == "adamw":
                return optim.AdamW([
                    {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': config.WEIGHT_DECAY},
                    {'params': head_params, 'lr': head_lr, 'weight_decay': config.WEIGHT_DECAY}
                ])
            else:
                return optim.SGD([
                    {'params': backbone_params, 'lr': backbone_lr, 'momentum': 0.9, 'weight_decay': config.WEIGHT_DECAY},
                    {'params': head_params, 'lr': head_lr, 'momentum': 0.9, 'weight_decay': config.WEIGHT_DECAY}
                ])
        else:
            # Phase 3: Same as Phase 2 but with SupCon
            if isinstance(model, nn.DataParallel):
                backbone_params = list(model.module.base_model.visual_encoder.parameters())
                head_params = list(model.module.classifier.parameters())
            else:
                backbone_params = list(model.base_model.visual_encoder.parameters())
                head_params = list(model.classifier.parameters())
            
            backbone_lr = 5e-7  # Even lower LR for Phase 3
            head_lr = 5e-5  # Slightly lower LR for head
            
            print(f"\n[Phase {phase}] Fine-tuning with SupCon!")
            print(f"  Backbone LR: {backbone_lr} | Head LR: {head_lr}")
            
            if config.OPTIMIZER == "adamw":
                return optim.AdamW([
                    {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': config.WEIGHT_DECAY},
                    {'params': head_params, 'lr': head_lr, 'weight_decay': config.WEIGHT_DECAY}
                ])
            else:
                return optim.SGD([
                    {'params': backbone_params, 'lr': backbone_lr, 'momentum': 0.9, 'weight_decay': config.WEIGHT_DECAY},
                    {'params': head_params, 'lr': head_lr, 'momentum': 0.9, 'weight_decay': config.WEIGHT_DECAY}
                ])
    
    # Initialize loss function - start with CE-only (Phase 1)
    def create_criterion_for_phase(phase):
        """Create loss function based on current phase"""
        if phase <= 2:
            return nn.CrossEntropyLoss()
        else:
            return CombinedLoss(temperature=config.TEMPERATURE)
    
    criterion = create_criterion_for_phase(current_phase)
    optimizer = create_optimizer_for_phase(current_phase, model, config)
    
    print(f"\n[Phase {current_phase}] Loss: {'CE-only' if current_phase <= 2 else 'Combined (SupCon + CE)'}")
    
    num_training_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = get_scheduler(optimizer, config, num_training_steps)
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if config.MIXED_PRECISION else None
    
    # Automatic multi-phase training loop
    print("\n" + "="*80)
    print("AUTOMATIC MULTI-PHASE TRAINING")
    print("="*80)
    print(f"Total epoch budget: {TOTAL_EPOCH_BUDGET}")
    print(f"Phase 1: Frozen backbone, CE-only (max {PHASE1_MAX_EPOCHS} epochs, target: {PHASE1_TARGET_ACC}% acc)")
    print(f"Phase 2: Unfrozen backbone, CE-only (max {PHASE2_MAX_EPOCHS} epochs, target: {PHASE2_TARGET_ACC}% acc)")
    print("Phase 3: Unfrozen backbone, SupCon+CE (remaining epochs up to total budget)")
    print(f"Batch size: {config.BATCH_SIZE} | Input size: {config.INPUT_SIZE}x{config.INPUT_SIZE}")
    print("="*80)
    
    best_val_loss = float('inf')
    global_epoch = 1
    phase1_start_epoch = 1
    phase2_start_epoch = None
    phase3_start_epoch = None
    
    while current_phase <= 3 and global_epoch <= TOTAL_EPOCH_BUDGET:
        # Determine max epochs for current phase
        if current_phase == 1:
            max_epochs = min(PHASE1_MAX_EPOCHS, TOTAL_EPOCH_BUDGET - global_epoch + 1)
            phase_start_epoch = phase1_start_epoch
        elif current_phase == 2:
            max_epochs = min(PHASE2_MAX_EPOCHS, TOTAL_EPOCH_BUDGET - global_epoch + 1)
            phase_start_epoch = phase2_start_epoch if phase2_start_epoch else global_epoch
        else:
            max_epochs = TOTAL_EPOCH_BUDGET - global_epoch + 1
            phase_start_epoch = phase3_start_epoch if phase3_start_epoch else global_epoch

        if max_epochs <= 0:
            break
        
        print(f"\n{'='*80}")
        print(f"PHASE {current_phase} - Epochs {phase_start_epoch} to {phase_start_epoch + max_epochs - 1}")
        print(f"{'='*80}")
        
        phase_best_acc = 0.0
        phase_completed = False  # Reset for each phase
        
        for epoch_in_phase in range(1, max_epochs + 1):
            epoch = global_epoch
            print(f"\n[Phase {current_phase}] Epoch {epoch_in_phase}/{max_epochs} (Global: {epoch})")
            print("-" * 80)
            
            # Train
            use_ce_only = (current_phase <= 2)
            train_loss, ce_loss, supcon_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, scaler,
                device, epoch, config, writer, use_ce_only=use_ce_only, num_epochs=max_epochs
            )
            
            print(f"Train Loss: {train_loss:.4f} | CE: {ce_loss:.4f} | SupCon: {supcon_loss:.4f} | Acc: {train_acc:.2f}%")
            
            # Track best accuracy in phase
            if train_acc > phase_best_acc:
                phase_best_acc = train_acc
            
            # Validate
            if epoch_in_phase % config.EVAL_EVERY_N_EPOCHS == 0 or epoch_in_phase <= 3:
                val_loss = validate(model, val_loader, criterion, device, epoch, config, writer, use_ce_only=use_ce_only)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"New best validation loss: {val_loss:.4f}")
                    
                    best_model_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_NAME)
                    save_checkpoint(
                        model, optimizer, epoch, best_val_loss,
                        best_model_path, is_best=True
                    )
            
            # Save checkpoint periodically
            if epoch % config.SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
                save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
                if current_phase == 2 or current_phase == 3:
                    backbone_lr = optimizer.param_groups[0]['lr']
                    head_lr = optimizer.param_groups[1]['lr']
                    writer.add_scalar('Train/LearningRate_Backbone', backbone_lr, epoch)
                    writer.add_scalar('Train/LearningRate_Head', head_lr, epoch)
                    print(f"Learning Rate: Backbone={backbone_lr:.2e} | Head={head_lr:.2e}")
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('Train/LearningRate', current_lr, epoch)
                    print(f"Learning Rate: {current_lr:.6f}")
            
            global_epoch += 1
            
            # Check phase transition conditions at end of epoch
            if AUTO_PHASE_TRAINING:
                should_transition = False
                
                if current_phase == 1:
                    # Phase 1 -> Phase 2: Check if target accuracy reached or max epochs
                    if train_acc >= PHASE1_TARGET_ACC or epoch_in_phase >= PHASE1_MAX_EPOCHS:
                        should_transition = True
                        phase_completed = True
                        if train_acc >= PHASE1_TARGET_ACC:
                            print(f"\n✅ Phase 1 Complete! Train accuracy {train_acc:.2f}% >= {PHASE1_TARGET_ACC}%")
                        else:
                            print(f"\n⏱️  Phase 1 Complete! Reached max epochs ({PHASE1_MAX_EPOCHS})")
                
                elif current_phase == 2:
                    # Phase 2 -> Phase 3: Check if target accuracy reached or max epochs
                    if train_acc >= PHASE2_TARGET_ACC or epoch_in_phase >= PHASE2_MAX_EPOCHS:
                        should_transition = True
                        # Don't set phase_completed here - we're transitioning, not completing Phase 3
                        if train_acc >= PHASE2_TARGET_ACC:
                            print(f"\n✅ Phase 2 Complete! Train accuracy {train_acc:.2f}% >= {PHASE2_TARGET_ACC}%")
                        else:
                            print(f"\n⏱️  Phase 2 Complete! Reached max epochs ({PHASE2_MAX_EPOCHS})")
                
                elif current_phase == 3:
                    # Phase 3: Check if the global 100-epoch budget has been reached.
                    if epoch_in_phase >= max_epochs:
                        phase_completed = True
                        print(f"\n✅ Phase 3 Complete! Reached total epoch budget ({TOTAL_EPOCH_BUDGET})")
                        # Don't break here - let the loop complete naturally
                
                # Perform phase transition if needed
                if should_transition:
                    if current_phase == 1:
                        # Transition to Phase 2
                        current_phase = 2
                        phase2_start_epoch = global_epoch
                        
                        # Unfreeze backbone
                        if isinstance(model, nn.DataParallel):
                            for param in model.module.base_model.visual_encoder.parameters():
                                param.requires_grad = True
                        else:
                            for param in model.base_model.visual_encoder.parameters():
                                param.requires_grad = True
                        
                        # Create new optimizer for Phase 2
                        optimizer = create_optimizer_for_phase(2, model, config)
                        criterion = create_criterion_for_phase(2)
                        num_training_steps = len(train_loader) * PHASE2_MAX_EPOCHS
                        scheduler = get_scheduler(optimizer, config, num_training_steps)
                        
                        print("Transitioning to Phase 2: Unfreezing backbone...")
                        break  # Exit inner loop, continue while loop with new phase
                    
                    elif current_phase == 2:
                        # Transition to Phase 3
                        current_phase = 3
                        phase3_start_epoch = global_epoch
                        
                        # Create new optimizer and criterion for Phase 3 (with SupCon)
                        optimizer = create_optimizer_for_phase(3, model, config)
                        criterion = create_criterion_for_phase(3)
                        remaining_epochs = max(TOTAL_EPOCH_BUDGET - global_epoch + 1, 1)
                        num_training_steps = len(train_loader) * remaining_epochs
                        scheduler = get_scheduler(optimizer, config, num_training_steps)
                        
                        print("Transitioning to Phase 3: Adding SupCon loss...")
                        break  # Exit inner loop, continue while loop with new phase
        
        # Check if Phase 3 completed (after inner loop finishes)
        # phase_completed is set inside the loop for Phase 3
        if current_phase == 3 and phase_completed:
            break
    
    print("\n" + "="*80)
    print("AUTOMATIC TRAINING COMPLETED!")
    print("="*80)
    print(f"Total epochs trained: {global_epoch - 1}")
    print(f"Phases completed: {current_phase}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_NAME)}")
    print("="*80)
    
    writer.close()


if __name__ == "__main__":
    main()
