1. Folder -> "checkpoints" contains:
    - A pretrained finetune-base model's weights.
    - 89% acc over test and train set for 7 Ninapro DB5 gestures:
        1. relax
        2. thumbs up
        3. flexion
        4. open hand
        5. fist
        6. wrist flexion
        7. wrist extension

2. Folder -> "finetuned":
    - Placeholder for a finetuned model's weights.

3. Folder -> "Ninapro_DB5" contains:
    - The Ninapro DB5 dataset I obtained from https://zenodo.org/record/1000116#.YraeZHaZNLE

4. Folder -> "tensorflow_lite" contains:
    - Placeholder for a compiled finetuned TF Lite model's weights.

5. Folder -> "visuals" contains:
    - Demonstration of how Myo Armband was weared when Ninapro DB5 collected their samples. (visuals\armband_position.png)
    - Overview of gestures and data samples: (visuals\Data.png + visuals\gestures.png)

6. File -> "config.py" contains:
    - Configuration Variables and Parameters

7. File -> "dataset.py" contains:
    - Utilities for extracting and preprocessing sEMG signals data

8. File -> "model.py" contains:
    - Code for A.I. model implementation and utility functions for it.

8. File -> "realtime.py" contains:
    - Main Program for Real-Time system which establishes BLE connection,
        defines GUI, and finetunes realtime samples from a pretrained finetune-base model.
    - OPTIONAL: Export finetuned model's weights

9. File -> "requirements.txt" contains:
    - List of libraries I used through this github repo
    - To replicate my environment, run this under activated conda env: pip install -r requirements.txt

10. File -> "scaling_params.json" contains: (If it exist)
    - Json with MEAN and Standard Deviation for each sensor Channel.

11. File -> "tensorflow_lite.py" contains:
    - Code for compiling Tensorflow model to Tensorflow Lite model.

12. File -> "train.py" contains:
    - A sample run of how to perform:
        - Data Preprocessing over Ninapro DataBase5
        - Training finetune-base model (Saving weights along the way)
        - Visualize training logs (model accuracy and loss during training)