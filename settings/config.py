from pathlib import Path

PROJECTS_PATH = Path().parent.resolve()

EXPORT_DIR = PROJECTS_PATH / "exported_model"

#################################################
# HYPER-PARAMETER OF ARCHITECTURE ###############
#################################################
ENC_STACK_SIZE = 5
DEC_STACK_SIZE = 5
NUM_HEADS = 8
D_MODEL = 512  # D_MODEL % NUM_HEADS == 0
D_FF = 2048
# MODALITIES - IMAGES
IMAGES_EMBEDDING_TYPE = 'ResNet-18'

#################################################
# BERT TOKENIZER ################################
#################################################
MODEL_TOKENIZER = 'bert-base-german-cased'
MAX_LENGTH_SEQUENCE = 32

# SHEDULE
LEARNING_RATE_TYPE = 'other'  # other
LEARCH_RATE = 1e-4

#################################################
# HYPER PARAMETER FOR TRAINING ##################
#################################################
EPOCHS = 10
NUM_SAMPLES = 55504
BATCH_SIZE = 2

SAVE_FREQ = 1000  # FREQUENCE SAVING CHECKPPOINT
