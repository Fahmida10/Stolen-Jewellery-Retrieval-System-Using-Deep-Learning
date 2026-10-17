"""
ResNet18 baseline configuration.
Single-phase transfer learning with cross-entropy supervision.
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
RUN_TAG = os.environ.get("RUN_TAG", "").strip()
RUN_STAGE_NAME = f"{STAGE_NAME}_{RUN_TAG}" if RUN_TAG else STAGE_NAME
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", RUN_TAG) if RUN_TAG else os.path.join(BASE_DIR, "checkpoints")
EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings", RUN_STAGE_NAME)
RESULTS_DIR = os.path.join(BASE_DIR, "results", RUN_STAGE_NAME)
BEST_MODEL_NAME = "best_model.pth"

BACKBONE = "resnet18"
NUM_CLASSES = 4

NUM_EPOCHS = 100
LEARNING_RATE = 0.1
BATCH_SIZE = 32
WEIGHT_DECAY = 0

FREEZE_BACKBONE = False
INPUT_SIZE = 224
USE_AUGMENTATION = False
NUM_WORKERS = 0
SEED = 42
