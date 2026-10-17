"""
Improved baseline configuration
Two-phase training (ResNet18), tuned LR, augmentation, mixed precision.
"""

import os

# Update this path before running on a new machine.
DATASET_ROOT = "/path/to/Jewellery Dataset Clean_copy"
TRAIN_DIR = os.path.join(DATASET_ROOT, "Train")
VAL_DIR = os.path.join(DATASET_ROOT, "Validation")
TEST_DIR = os.path.join(DATASET_ROOT, "Test")
STOLEN_DIR = os.path.join(DATASET_ROOT, "stolen")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_NAME = os.path.basename(BASE_DIR)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BEST_MODEL_NAME = "best_model.pth"

BACKBONE = "resnet18"
NUM_CLASSES = 4

PHASE1_EPOCHS = 30
PHASE2_EPOCHS = 70
PHASE1_LR = 1e-3
PHASE2_BACKBONE_LR = 1e-5
PHASE2_HEAD_LR = 1e-4

BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4

INPUT_SIZE = 224
USE_AUGMENTATION = True
NUM_WORKERS = 0
SEED = 42

MIXED_PRECISION = True
USE_SCHEDULER = True
