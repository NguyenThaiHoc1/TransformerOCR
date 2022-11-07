from pathlib import Path

PROJECTS_PATH = Path().parent.resolve()

EXPORT_DIR = PROJECTS_PATH / "exported_model"

# SHAPE IMAHGE STATIC
INPUT_SHAPE = (150, 600, 3)

# BERT TOKENIZER
MODEL_TOKENIZER = 'bert-base-german-cased'
MAX_LENGTH_SEQUENCE = 32

# SHEDULE
LEARNING_RATE_TYPE = 'other'  # other
LEARCH_RATE = 1e-4

# EMBEDDING SIZE
MODEL_SIZE = 256

# MODALITIES - IMAGES
IMAGES_EMBEDDING_TYPE = 'InceptionV3'

# HYPER PARAMETER FOR TRAINING
EPOCHS = 10
NUM_SAMPLES = 308
BATCH_SIZE = 4

SAVE_FREQ = 20