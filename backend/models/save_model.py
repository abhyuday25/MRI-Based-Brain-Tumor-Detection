import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D
)

def create_and_save_model(save_path: str):
    """
    Creates the CNN model structure as defined in TumorML.ipynb 
    and saves it to the specified path.
    """
    model = Sequential()

    # Layer 1
    model.add(Conv2D(64, (5,5), padding='same', activation='relu',
                     input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # Layer 2
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # Layer 3
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.30))

    # Layer 4
    model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.30))

    # Fully Connected
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    # Save the model structure
    model.save(save_path)
    print(f"Model architecture saved to {save_path}")

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "classification.keras")
    create_and_save_model(output_path)
