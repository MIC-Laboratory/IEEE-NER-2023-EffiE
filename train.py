"""
    Description:
        Achieves:
            - Data Preprocessing over Ninapro DataBase5
            - Training finetune-base model (Saving weights along the way)
            - Visualize training logs (model accuracy and loss during training)
            
    Author: Jimmy L. @ SF State MIC Lab
    Date: Summer 2022
"""
from dataset import *
from model import *
import config
import tensorflow as tf



if __name__ == "__main__":
    # NOTE: Check if Utilizing GPU device
    print(tf.config.list_physical_devices('GPU'))
    
    # NOTE: Data Preprocessings
    
    # Get sEMG samples and labels. (shape: [num samples, 8(sensors/channels)])
    emg, label = folder_extract(
        config.folder_path,
        exercises=config.exercises,
        myo_pref=config.myo_pref
    )
    # Apply Standarization to data, save collected MEAN and STANDARD DEVIATION in the dataset to json
    emg = standarization(emg, config.std_mean_path)
    # Extract sEMG signals for wanted gestures.
    gest = gestures(emg, label, targets=config.targets)
    # Perform train test split
    train_gestures, test_gestures = train_test_split(gest)
    
    # NOTE: optional visualization that graphs class/gesture distributions
    # plot_distribution(train_gestures)
    # plot_distribution(test_gestures)
    
    # Convert sEMG data to signal images.
    X_train, y_train = apply_window(train_gestures, window=config.window, step=config.step)
    # Convert sEMG data to signal images.
    X_test, y_test = apply_window(test_gestures, window=config.window, step=config.step)
    
    X_train = X_train.reshape(-1, 1, 8, 52)
    X_test = X_test.reshape(-1, 1, 8, 52)
    
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    
    # Get Tensorflow model
    cnn = get_model(
        num_classes=config.num_classes, # 4
        filters=config.filters, # [32, 64]
        neurons=config.neurons, # [512, 128]
        dropout=config.dropout # 0.5
    )
    
    # Start training (And saving weights along training)
    history = train_model(
        cnn, X_train, y_train, X_test, y_test,
        config.batch_size, save_path=config.save_path, epochs=config.epochs,
        patience=config.patience, lr=config.inital_lr
    )
    
    # Visualize accuarcy and loss logs
    plot_logs(history, acc=True, save_path=config.acc_log)
    plot_logs(history, acc=False, save_path=config.loss_log)
    
    # Load pretrained model
    model = get_model(
        num_classes=config.num_classes, # 4
        filters=config.filters, # [32, 64]
        neurons=config.neurons, # [512, 128]
        dropout=config.dropout # 0.5
    )
    model.load_weights(config.save_path)
    
    # # NOTE: Optional test for loaded model's performance
    # model.compile(
    #         optimizer=tf.keras.optimizers.Adam(learning_rate=0.2),
    #         loss='sparse_categorical_crossentropy',
    #         metrics=['accuracy'],
    #     )
    # # See if weights were the same
    # model.evaluate(X_test, y_test)
    
    # Test with finetune model. (last classifier block removed from base model)
    finetune_model = get_finetune(config.save_path, config.prev_params)
    
    # NOTE: You can load finetune model like this too.
    # finetune_model = create_finetune(model, num_classes=4)
    # finetune_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.2),
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy'],
    # )
    # finetune_model.evaluate(X_test, y_test)