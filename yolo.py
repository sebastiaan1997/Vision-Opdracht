from re import X
from keras.models import Sequential, Model


from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Conv3D, Conv2D, MaxPooling2D, Dense, Flatten, GlobalAveragePooling2D, Dropout, Input, Resizing, Rescaling, Layer
from keras.utils import to_categorical, load_img, img_to_array, image_dataset_from_directory, plot_model
from keras.utils.image_dataset import paths_and_labels_to_dataset
import numpy as np
import tensorflow as tf
from pathlib import Path


def yolo_leaky_activation(x) -> float:
    """Activation of YoLo"""
    if x > 0:
        return X
    else:
        return 0.1 * x


def get_yolo_model() -> Model:
    """Implementatie van het YoLo CNN zoals beschreven in paper"""
    model = Conv2D(64, (7, 7), strides=(2, 2),
                   activation=yolo_leaky_activation)
    model = MaxPooling2D((2, 2), strides=(2, 2))(model)
    model = Conv2D(192, (3, 3), activation=yolo_leaky_activation)(model)
    model = MaxPooling2D((2, 2), strides=(2, 2))(model)
    model = Conv2D(128, (1, 1), activation=yolo_leaky_activation)(model)
    model = Conv2D(256, (3, 3), activation=yolo_leaky_activation)(model)
    model = Conv2D(256, (1, 1), activation=yolo_leaky_activation)(model)
    model = Conv2D(512, (3, 3), activation=yolo_leaky_activation)(model)
    model = MaxPooling2D((2, 2), strides=(2, 2))(model)
    model = Conv2D(256, (1, 1), activation=yolo_leaky_activation)(model)
    model = Conv2D(512, (3, 3), activation=yolo_leaky_activation)(model)
    model = Conv2D(256, (1, 1), activation=yolo_leaky_activation)(model)
    model = Conv2D(512, (3, 3), activation=yolo_leaky_activation)(model)
    model = Conv2D(1024, (3, 3), activation=yolo_leaky_activation,
                   strides=(2, 2))(model)
    model = Conv2D(1024, (3, 3), activation=yolo_leaky_activation)(model)
    model = Dense(1024, activation=yolo_leaky_activation)(model)
    model = Dense(30, activation=yolo_leaky_activation)(model)
    return Model(outputs=[model])


if __name__ == "__main__":
    get_yolo_model()
