## [IEEE NER 2023 - EffiE: Efficient Convolutional Neural Network for Real-Time EMG Pattern Recognition System on Edge Devices](https://ieeexplore.ieee.org/document/10123741/)

![Fine-Tuning](visuals/finetune.png?raw=true "Fine-Tuning")
* Figure 1: Convolutional Neural Network and Fine-Tuning

## About
This project aims to achieve efficient sEMG gesture recognition through Convolutional Neural Network (CNN) and Finetuning. Finetuning is a technique of transfer learning, which, we aim to generalize model learnings from the larger dataset to a more specific, downstream dataset. In this case, we performed offline training with a large recorded sEMG dataset and retrained it with real-time collected data. Additionally, we compiled our model with Tensorflow Lite, optimized for micro-controller applications.

## Data Collections
To acquire real-time sEMG signals, we utilized Thalmic Lab's Myo Armband, with BLE connections established via. the python Bleak library. The Myo Armband contains 8 channels of raw sEMG signal sampled at 200 Hz. As for offline training data, we adopted the NinaPro DB5, which also uses Myo Armband for data acquisition. From NinaPro DB5, we utilized 7 gestures collected using the Myo Armband closest to elbow from Exercise B: 

    1. Rest (0)
    2. Thumb Up (13)
    3. Flexion of ring and little finger, extension of the others (15)
    4. Abduction of all fingers (17)
    5. Fingers flexed together in fist (18)
    6. Wrist flexion (25)
    7. Wrist Extension (26)

## Quick Start

#### Real Time Gesture Recognition
Let's perform a simple 4 gesture recognition. Enable bluetooth on your PC and have the Myo Armband attached on your arm. Then, run `realtime.py` and follow guidelines printed on the command prompt, to perform the 4 gestures: "Rest", "Fist", "Thumbs Up", and "Ok Sign". After real time sEMG is collected, the pre-trained model from folder "checkpoints" will be finetuned with these samples. Lastly, this model will perform gesture recognition! (Optional: You may also export the finetuned model's weights by controlling variable "finetuned_path")

#### Advanced
If you would like to perform other gestures, you may also finetune with other gestures from NinaPro DB5, simply by controlling the variable "targets" in `config.py`. Then, edit the "GESTURES" in `realtime.py` for realtime gesture recognition upon other gestures.

#### Microcontroller Section
Please refer to this GitHub repo for the prosthesis arm control implementation on microcontrollers: https://github.com/MIC-Laboratory/Real-time-Bionic-Arm-Control-via-CNN-on-Sony-Spresense


![sEMG Data Preprocessing](visuals/data_preprocessing.png?raw=true "sEMG Data Preprocessing")
* Figure 2: sEMG Data Preprocessing

## Data Preprocessing
As shown in the fig.2, the Myo Armband includes 8 channels, so each raw sEMG value is an 8-bit unsigned number ranging 0 - 255, so we change the sEMG values from unsigned to signed by subtracting the value by 256 if the sEMG value is greater than 127. One sEMG sample per channel sampled at 5 ms makes up an sEMG array shaped 1x8. Then, we combine 32 sEMG arrays to create an sEMG window (8 × 32). So, each window is sampled at 160 ms with 80 ms overlapping step of 16. Finally, we will have one set of mean and standard deviation for each of the 8 channels calculated from the sEMG samples, obtained from the NinaPro DB5 dataset gestures used during pre-training. And we subtract each EMG value with the local mean divided by the local standard deviation. File `dataset.py` contains all data preprocessing related code, and file `train.py` lines 19 - 50 performs sample test run.

## Offline Training
According to Wikipedia, Convolutional Neural Network is a class of artificial neural network, most commonly applied to analyze visual imagery. In this case, our "image" will be the 8x32 sEMG window. Therefore, the model input shape is [batch size, 8, 32, 1], batch_size x height x length x channels/depth. (would be 3 if image was RGB format) Our CNN model consists of 2 convolutional layers followed by a fully-connected layer. The first convolutional layer consists of 32-filters, followed by a PReLU activation function, batch normalization to accelerate model training, spatial 2D dropout to counter overfitting, and max pooling. The second convolutional layer is the same as the first layer except that it uses 64 filters instead of 32. Lastly, we will end with N neuron fully-connected layer depending on N gestures utilized during pre-training. File `model.py` contains all utility code regarding CNN, and file `train.py` lines 64 - 79 performs sample test run. To control what gestures to pre-train with from NinaPro DB5, simply edit variable "targets" in `config.py`. (Other hyperparameters also in `config.py`)

###### Special NOTE
To run realtime gesture recognition, follow the "Real Time Gesture Recognition" and "Advanced" section above.

## Tensorflow Lite (optional)
Have the finetuned model's weights stored in folder `finetuned`, and run `tensorflow_lite.py`. Then, you will see the optimized TF-Lite model in folder `tensorflow_lite`.

## Files Overview
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

6. `config.py` contains:
    - Configuration Variables and Parameters

7. `dataset.py` contains:
    - Utilities for extracting and preprocessing sEMG signals data

8. `model.py` contains:
    - Code for A.I. model implementation and utility functions for it.

8. `realtime.py` contains:
    - Main Program for Real-Time system which establishes BLE connection,
        defines GUI, and finetunes realtime samples from a pretrained finetune-base model.
    - OPTIONAL: Export finetuned model's weights

9. `requirements.txt` contains:
    - List of libraries I used through this github repo
    - To replicate my environment, run this under activated conda env: pip install -r requirements.txt

10. `scaling_params.json` contains: (If it exist)
    - Json with MEAN and Standard Deviation for each sensor Channel.

11. `tensorflow_lite.py` contains:
    - Code for compiling Tensorflow model to Tensorflow Lite model.

12. `train.py` contains:
    - A sample run of how to perform:
        - Data Preprocessing over Ninapro DataBase5
        - Training finetune-base model (Saving weights along the way)
        - Visualize training logs (model accuracy and loss during training)
