import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, Dense
import matplotlib.pyplot as plt

DATASET_PATH = "Data.json"

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)
    
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

if __name__ == "__main__":
    # Load dataset
    inputs, targets = load_data(DATASET_PATH)

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3)

    # Define the model
    model = keras.Sequential([
        # Input layer
        Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        # Hidden layers
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(64, activation='relu'),
        # Output layer
        Dense(len(np.unique(targets)), activation='softmax')
    ])

    
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)

    # Compile the model
    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Model Summary
    model.summary()

    # Train the model
    history = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test))

    # Plot accuracy and error over the epcohs
    plot_history(history)
