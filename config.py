# TRAINER SETTINGS
BATCH_SIZE = 128
EPOCHS = 10
NUM_WORKERS = 4

# OPTIMIZER SETTINGS
LR = 0.001
GRAD_NORM_CLIP = 0.5

# MODEL HPARAMS
N_CLASSES = 10
N_INPUT = 28  # WE ASSUME SQUARE PICTURES
N_HIDDEN = 64
N_HIDDEN_CNN = 64
N_HIDDEN_LIN = 64
KERNEL_SIZE = 5

MAX_STEPS = 20
LAMBDA_P = 0.2
BETA = 0.01