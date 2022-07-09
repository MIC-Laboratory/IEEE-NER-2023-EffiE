"""
    Description: Configuration Variables and Parameters
    Author: Jimmy L. @ SF State MIC Lab
    Date: Summer 2022
"""
# How many samples each sEMG image channel contains.
window = 52

# Sliding step (for overlapping)
step = 5

# Exercises with dedicated gestures stored
exercises = ["E2"]

# Path of Ninapro DataBase 5 sEMG dataset.
folder_path = "Ninapro_DB5"

# Ninapro DB5 data collected via 2 Myo armband, controls which armband's 8 sensors to collect
myo_pref = "elbow"

# Class of gestures for training finetune-base model.
targets = [0, 1, 3, 6]

# Path to save fintune-base model. (Make sure ends with '.ckpt').
save_path = "checkpoints/model.ckpt"

# Number of gestures to detect for finetune-base model.
num_classes = 4

# Number of gestures to detect for finetuned model.
fine_tune_classes = 4

# Number of CNN output filters the model contains.
filters = [32, 64]

# Number of neurons for FFN the model contains.
neurons = [512, 128]

# Dropout rate.
dropout = 0.5

# The number of epochs without improvement after which training will be early stopped
patience = 80

# Initial learning rate for training finetune-base model.
inital_lr = 0.2

# Number of training epochs for the finetune-base model.
epochs = 300

# Batch size for training the finetune-base model.
batch_size = 384 # 256

# Paths for saving logs generated when training finetune-base model.
acc_log = 'visuals/acc_log.jpg'
loss_log = 'visuals/loss_log.jpg'

# Params info needed to load pretrained finetune-base model.
prev_params = [num_classes, filters, neurons, dropout]

# Path of json with MEAN and Standard Deviation for each sensor Channel.
std_mean_path = "scaling_params.json"