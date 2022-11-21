from pathlib import Path

PROJECTS_PATH = Path().parent.resolve()

EXPORT_DIR = PROJECTS_PATH / "exported_model"

# SHAPE IMAHGE STATIC
INPUT_SHAPE = (32, 768, 3)

# BERT TOKENIZER
MODEL_TOKENIZER = 'bert-base-german-cased'
MAX_LENGTH_SEQUENCE = 32

# SHEDULE
LEARNING_RATE_TYPE = 'other'  # other
LEARCH_RATE = 1e-4

# EMBEDDING SIZE
MODEL_SIZE = 512

# MODALITIES - IMAGES
IMAGES_EMBEDDING_TYPE = 'ResNet-18'

# HYPER PARAMETER FOR TRAINING
# DATA-TRAINING
EPOCHS = 10
NUM_SAMPLES = 55504
BATCH_SIZE = 2

# DATA-VALIDATE


# FREQUENCE SAVING CHECKPPOINT
SAVE_FREQ = 1000
