"""OpenCLIP ViT-B/32 baseline configuration."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_NAME = os.path.basename(BASE_DIR)
# Update this path before running on a new machine.
DATASET_ROOT = "/path/to/Jewellery Dataset Clean_copy"
TRAIN_DIR = os.path.join(DATASET_ROOT, "Train")
VAL_DIR = os.path.join(DATASET_ROOT, "Validation")
TEST_DIR = os.path.join(DATASET_ROOT, "Test")
STOLEN_DIR = os.path.join(DATASET_ROOT, "stolen")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BEST_MODEL_NAME = "best_model.pth"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
EMBEDDING_DIM = 512

INPUT_SIZE = 224
NORMALIZE_EMBEDDINGS = True
NUM_CLASSES = 4

NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4

USE_AUGMENTATION = True
NUM_WORKERS = 0
MIXED_PRECISION = True
SEED = 42
