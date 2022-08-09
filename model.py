"""
    Description: Code for A.I. model implementation and utility functions for it.
    Author: Jimmy L. @ SF State MIC Lab
    Date: Summer 2022
"""
from tabnanny import verbose
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
tf.get_logger().setLevel('INFO')


def get_model(num_classes=4, filters=[32, 64], neurons=None, dropout=0.5,
              kernel_size=(5, 3), input_shape=(52, 8, 1), pool_size=(3, 1)):
    assert len(filters) == 2
    """
    Purpose:
        Establish the architecture for the finetune-base A.I. model.

    Args:
        1. num_classes (int, optional):
            Number of classes/gestures to classify. Defaults to 4.
            
        2. filters (1D list, optional):
            A list specifying number of output filters for the first and second 2D CNN. Defaults to [32, 64].
            
        3. neurons (1D list, optional):
            A list specifying number of neurons for the first and second neural network. Defaults to None.
            
        4. dropout (float, optional):
            Dropout rate. Defaults to 0.5.
        
        5. kernel_size (tuple):
            kernel window size for CNN. Defaults to (3, 5)
        
        6. input_shape (tuple):
            Input shape for CNN. Defaults to (8, 52, 1) channel LAST

    Returns:
        1. model (keras.engine.sequential.Sequential):
            - The finetune-base model takes inputs of shape:
                    
                    [batch_size, 1, 8, 52]
                    
                - batch_size is batch_size
                - 1 refers to input channels. (like 3 from RGB images)
                - 8 refers to number of Myo armband sensors/channels (vertical width)
                - 52 refers to window size, how many samples included per sensor/channel (horizontal length)
    """

    CNN1 = tf.keras.layers.Conv2D(
            filters=filters[0],
            strides=1,
            kernel_size=kernel_size, # 3x5 window
            activation='relu',
            input_shape=input_shape
        )
    CNN2 = tf.keras.layers.Conv2D(
        filters=filters[1],
        strides=1,
        kernel_size=kernel_size, # 3x5 window
        activation='relu'
    )

    model = tf.keras.Sequential([
        # """
        # First CNN Feature Extraction Block
        # """
        CNN1,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.PReLU(),
        tf.keras.layers.SpatialDropout2D(rate=dropout),
        tf.keras.layers.MaxPool2D(pool_size=pool_size),
        # """
        # Second CNN Feature Extraction Block
        # """
        CNN2,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.PReLU(),
        tf.keras.layers.SpatialDropout2D(rate=dropout),
        tf.keras.layers.MaxPool2D(pool_size=pool_size),
        
        tf.keras.layers.Flatten()
    ])
    if neurons != None:
        for ffn_size in neurons:
            model.add(tf.keras.layers.Dense(ffn_size))
            model.add(tf.keras.layers.PReLU())
            
    # """
    # Last Forward Neural Network (Classifier Block)
    # """
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Softmax(axis=-1))
    
    return model


def create_finetune(base_model, num_classes=4):
    """
    Purpose:
        Generate a new finetune model from the pretrained finetune-base model
        NOTE: Last neural net block of the 'base_model'(from args) replaced a new block of 'num_classes'(from args) neurons

    Args:
        1. base_model (keras.engine.sequential.Sequential):
            The pretrained finetune-base model.
            
        2. num_classes (int, optional):
            Number of gestures/classes the finetune model would like to classify. Defaults to 4.

    Returns:
        1. new_model (keras.engine.sequential.Sequential):
            - The new finetune model with majority architecture derived from the 'base_model'(from args)
            - The finetune model takes inputs of shape:
                    
                    [batch_size, 8, 52, 1]
                    
                - batch_size is batch_size
                - 1 refers to input channels. (like 3 from RGB images)
                - 8 refers to number of Myo armband sensors/channels (vertical width)
                - 52 refers to window size, how many samples included per sensor/channel (horizontal length)
    """
    new_model = tf.keras.Sequential()
    # Append through until last 2 layers (classifier block + softmax layer)s
    for layer in base_model.layers[:-2]:
        new_model.add(layer)

    # Add new blocks of output classifier neural net.
    new_model.add(tf.keras.layers.Dense(num_classes))
    new_model.add(tf.keras.layers.Softmax(axis=-1))
    return new_model


def get_pretrained(path, prev_params):
    base_model = get_model(
        num_classes=prev_params[0], # 4
        filters=prev_params[1], # [32, 64]
        neurons=prev_params[2], # [512, 128]
        dropout=prev_params[3], # 0.5
        kernel_size=prev_params[4],
        input_shape=prev_params[5],
        pool_size=prev_params[6]
    )
    # Load pretrained weights
    base_model.load_weights(path).expect_partial()
    base_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return base_model


def get_finetune(path, prev_params, lr=0.0001, num_classes=4):
    """
    Purpose:
        Direct return a new finetune-model, with finetune-base model loaded with 'path'(from args).

    Args:
        1. path (str):
            - Path of pretrained weights of finetune-base model
        
        2. prev_params (list):
            - Parameters specification of the pretrained finetune-base model
        
        3. lr (float, optional):
            - Learning rate for the new finetune model (recommend setting small learning rate). Defaults to 0.0001.
            
        4. num_classes (int, optional):
            - Number of gestures/classes the new finetune model would like to classify. Defaults to 4.

    Returns:
        1. finetune_model (keras.engine.sequential.Sequential):
            - The new finetune model with majority architecture derived from the 'base_model'(from args)
            - The finetune model takes inputs of shape:
                    
                    [batch_size, 1, 8, 52]
                    
                - batch_size is batch_size
                - 1 refers to input channels. (like 3 from RGB images)
                - 8 refers to number of Myo armband sensors/channels (vertical width)
                - 52 refers to window size, how many samples included per sensor/channel (horizontal length)
    """
    # Get architecture of finetune-base model
    base_model = get_model(
        num_classes=prev_params[0], # 4
        filters=prev_params[1], # [32, 64]
        neurons=prev_params[2], # [512, 128]
        dropout=prev_params[3], # 0.5
        kernel_size=prev_params[4],
        input_shape=prev_params[5],
        pool_size=prev_params[6]
    )
    # Load pretrained weights
    base_model.load_weights(path).expect_partial()
    
    # Create finetune model
    finetune_model = create_finetune(base_model, num_classes=num_classes)
    
    # Compile finetune model with optimizer, loss funcs, eval metrics
    finetune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    return finetune_model


def train_model(model, X_train, y_train, X_test, y_test, batch_size,
                save_path=None, epochs=200, patience=80, lr=0.2, decay_rate=0.9):
    """
    Purpose:
        Train the finetune-base model

    Args:
        1. model (keras.engine.sequential.Sequential):
            The finetune-base model to train
        
        2. X_train (numpy.ndarray):
            The training input. Shape: [number of samples, 1, 8(sensors/channels), 52(window size)]
        
        3. y_train (numpy.ndarray):
            The training target/label. Shape: [number of samples]
        
        4. X_test (numpy.ndarray):
            The testing input. Shape: [number of samples, 1, 8(sensors/channels), 52(window size)]
        
        5. y_test (numpy.ndarray):
            The testing target/label. Shape: [number of samples]
        
        6. batch_size (int):
            Batch_size for training the finetune-base model
        
        7. save_path (str):
            Path to save the finetune-base model's weights. (Should end with '.ckpt').
        
        8. epochs (int, optional):
            Number of training epochs. Defaults to 200.
            
        9. patience (int, optional):
            The number of epochs without improvement after which training will be early stopped. Defaults to 80.
             
        10. lr (float, optional):
            Initial learning rate for training the finetune-base model. Defaults to 0.2.
            
        11. decay_rate (float, optional):
            Decay rate of learning rate scheduler. Defaults to 0.9.

    Returns:
        1. history (keras.callbacks.History):
            History log of training loss and accuracies.
            
    Additional Note: Use .save_weights(f"{name}.ckpt") to replicate this
    """
    callback_lists = []
    
    # Save model weights to 'save_path'(from args) if provided.
    if save_path != None:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            save_path, monitor='val_loss', verbose=1, save_freq='epoch',
            save_best_only=True, mode='min', save_weights_only=True
        )
        callback_lists.append(checkpoint)
    
    # Add early stopping
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=patience
    )
    callback_lists.append(early)
    
    # Get learning rate scheduler.
    decay_steps = (len(X_train) / batch_size) * 1.5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )
    
    # Compile model with optimizer, loss funcs, eval metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    # Model fitting
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callback_lists
    )
    
    return history


def plot_logs(history, acc=True, save_path=None):
    """
    Purpose:
        Plot loss and accuracy logs from model training.

    Args:
        1. history (keras.callbacks.History):
            The loss and accuracy log output from model training
            
        2. acc (bool, optional):
            Whether to plot training accurcy logs. Defaults to True. (False -> plot loss logs)
        
        3. save_path (str, optional):
            Path to save plot. (Should end with '.jpg') Defaults to None.
    """
    if acc == True:
        params = ["accuracy", "val_accuracy", "model accuracy", "accuracy"]
    else:
        params = ["loss", "val_loss", "model loss", "loss"]
    
    plt.figure(figsize=(20, 6))
    plt.plot(history.history[params[0]])
    plt.plot(history.history[params[1]])
    plt.title(params[2])
    plt.ylabel(params[3])
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(save_path)
    plt.show()

tf.get_logger().setLevel('INFO')
def realtime_pred(model, sEMG, num_channels=8, window_length=32):
    """
    Purpose:
        Perform realtime predictions with the finetuned model.
    
    Args:
        1. model (keras.engine.sequential.Sequential):
            The finetuned model
            
        2. sEMG (numpy.ndarray):
            The realtime sEMG samples to input
            
        3. num_channels (int, optional):
            Number of Myo Armband sensors/channels. Defaults to 8.
            
        4. window_length (int, optional):
            How many samples included per sensor/channel (horizontal length). Defaults to 52.

    Returns:
        (numpy.int64):
            The model prediction index
    """
    # Reshape sample to proper sEMG image
    sEMG = np.array(sEMG).reshape(-1, num_channels, window_length, 1)
    # Run model predictions
    pred = model.predict(sEMG, verbose=0)
    # Return location/index of maximum prediction value
    return np.argmax(pred)


#########
# # NOTE: Tensorflow model implementation with sub-class method.
#########

# class Model(tf.keras.Model):
#   def __init__(self, num_classes=4, filters=[32, 64], neurons=[512, 128], dropout=0.5):
#     super(Model, self).__init__()
#     self.conv_set1 = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(
#             filters=filters[0],
#             strides=1,
#             kernel_size=(3, 5), # 3x5 window
#             activation='relu',
#             input_shape=(1, 8, 52),
#             data_format="channels_first"
#         ),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.PReLU()
#     ])
    
#     self.dropout1 = tf.keras.layers.SpatialDropout2D(rate=dropout, data_format='channels_first')
#     self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(1, 3), data_format='channels_first') # 1x3 window
    
#     self.conv_set2 = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(
#             filters=filters[1],
#             strides=1,
#             kernel_size=(3, 5), # 3x5 window
#             activation='relu',
#             data_format="channels_first"
#         ),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.PReLU()
#     ])
    
#     self.dropout2 = tf.keras.layers.SpatialDropout2D(rate=dropout, data_format='channels_first')
#     self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(1, 3), data_format='channels_first') # 1x3 window
    
#     self.flatten = tf.keras.layers.Flatten()
    
#     self.ffn = tf.keras.Sequential([
#         tf.keras.layers.Dense(neurons[0]),
#         tf.keras.layers.PReLU(),
#         tf.keras.layers.Dense(neurons[1]),
#         tf.keras.layers.PReLU(),
#     ])
        
#     self.classifier = tf.keras.Sequential([
#         tf.keras.layers.Dense(num_classes),
#         tf.keras.layers.Softmax(axis=-1)
#     ])
    
#   def call(self, inputs):
#     out = self.conv_set1(inputs)
#     out = self.dropout1(out)
#     out = self.maxpool1(out)
    
#     out = self.conv_set2(inputs)
#     out = self.dropout2(out)
#     out = self.maxpool2(out)
    
#     out = self.flatten(out)

#     out = self.ffn(out)
    
#     out = self.classifier(out)
    
#     return out