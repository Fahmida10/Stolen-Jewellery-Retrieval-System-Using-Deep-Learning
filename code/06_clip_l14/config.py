"""
Configuration for Jewelry Retrieval System - OpenCLIP ViT-L/14
"""

import os

class Config:
    # Update this path before running on a new machine.
    DATASET_ROOT = "/path/to/Jewellery Dataset Clean_copy"
    TRAIN_DIR = os.path.join(DATASET_ROOT, "Train")
    VAL_DIR = os.path.join(DATASET_ROOT, "Validation")
    TEST_DIR = os.path.join(DATASET_ROOT, "Test")
    STOLEN_DIR = os.path.join(DATASET_ROOT, "stolen")

    _BASE = os.path.dirname(os.path.abspath(__file__))
    STAGE_NAME = os.path.basename(_BASE)
    PROJECT_ROOT = _BASE
    CHECKPOINT_DIR = os.path.join(_BASE, "checkpoints")
    EMBEDDING_DIR = os.path.join(_BASE, "embeddings")
    RESULTS_DIR = os.path.join(_BASE, "results")

    MODEL_NAME = "ViT-L-14"
    PRETRAINED = "openai"
    EMBEDDING_DIM = 768

    INPUT_SIZE = 224
    NORMALIZE_EMBEDDINGS = True

    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5

    LOSS_TYPE = "supcon"
    TEMPERATURE = 0.07
    MARGIN = 0.2
    HARD_MINING = True

    OPTIMIZER = "adamw"
    SCHEDULER = "cosine"

    USE_AUGMENTATION = True
    COLOR_JITTER = True
    RANDOM_ROTATION = 15

    SIMILARITY_METRIC = "cosine"
    TOP_K = 10

    NUM_WORKERS = 0
    PIN_MEMORY = True
    MIXED_PRECISION = True

    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5
    BEST_MODEL_NAME = "best_model.pth"

    DEVICE = "cuda"
    EVAL_EVERY_N_EPOCHS = 5
    SEED = 42
