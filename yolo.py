from email.mime import base
from waldo import get_waldos, load_images
import cv2
import itertools as itr
from typing import Tuple, List, Iterable
from math import sqrt
from pathlib import Path
import tensorflow as tf
import numpy as np
from keras.utils.image_dataset import paths_and_labels_to_dataset
from keras.utils import to_categorical, load_img, img_to_array, image_dataset_from_directory, plot_model
from keras.layers import Conv3D, Conv2D, MaxPooling2D, Dense, Flatten, GlobalAveragePooling2D, Dropout, Input, Resizing, Rescaling, Layer, Reshape, LeakyReLU
from keras.optimizers import Adam
from keras.backend import less
from keras.models import Sequential, Model
from re import X
from keras.regularizers import l2
import keras.backend as k
import os
import datetime
from keras.callbacks import LearningRateScheduler

from json import dumps

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=2000)])
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# from cython import compile


class YoloReshape(Layer):
    def __init__(self, target_shape, class_amount: int):
        super(YoloReshape, self).__init__()
        self.target_shape = target_shape
        self.class_amount = class_amount

    def get_config(self):
        """"""
        conf = super().get_config().copy()
        conf.update({
            "target_shape": self.target_shape
        })
        return conf

    def call(self, input):
        pass

    # def yolo_loss(y_true, y_pred):
    # """Loss function implemented according to the yolo paper"""
    # pos_loss=[]
    # size_loss=[]
    # for true, pred in zip(y_true, y_pred):
    # t_x, t_y, t_w, t_h=true
    # p_x, p_y, p_w, p_h=pred
    # pos_loss.append((t_x - p_x ** 2) + (t_y + p_y) ** 2)
    # size_loss.append(((sqrt(p_w) - sqrt(t_w)) +
    #                   (sqrt(p_h) - sqrt(t_h))) ** 2)

    # def yolo_loss(y_true, y_pred):
    # len(y_true

# return sum(pos_loss) + sum(size_loss)


def get_base_index(cell: int) -> int:
    return 1 + (5*cell)


def yolo_position_loss(prediction, truth, cell: int):
    base_index = get_base_index(cell)
    # Index 0 is x coordinate
    # Index 1 is y coordinate
    return (prediction[base_index + 1] - truth[base_index + 1]) ** 2 + (prediction[base_index + 2] - truth[base_index + 2])


def yolo_size_loss(prediction, truth, cell: int):
    base_index = get_base_index(cell)
    return (sqrt(prediction[base_index + 3]) - sqrt(truth[base_index + 3])) ** 2 + (sqrt(prediction[base_index + 4]) - sqrt(truth[base_index + 4]))


def object_in_cell(prediction, truth, cell: int) -> bool:
    base_index = get_base_index(cell)
    return truth[base_index + 0] >= 1.0


def classification_loss(prediction, truth, cell: int) -> float:
    base_index = get_base_index(cell)
    return (truth[base_index + 0] - prediction[base_index + 0]) ** 2


def classification_loss(prediction, truth, cell: int) -> float:
    return (truth[0] - prediction[0]) ** 2


@tf.function
def yolo_loss(y_true, y_pred):
    loss = 0.0
    s = tf.shape(y_true)
    print(s)

    for x in s[0]:
        pred = y_pred[x]
        true = y_true[x]
        for i in range(9):
            if object(pred, true, i):
                loss += yolo_position_loss(pred, true, i) + yolo_size_loss(pred, true, i) + \
                    classification_loss(pred, true, i)
            else:
                loss += classification_loss(pred, true, i)
    return loss


yolo_leaky_activation = "relu"


def get_yolo_model(img_h=448, img_w=448) -> Model:
    """Implementatie van het YoLo CNN zoals beschreven in paper"""
    lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), input_shape=(
        img_h, img_w, 3), padding='same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=192, kernel_size=(3, 3), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=128, kernel_size=(1, 1), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(1, 1), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=512, kernel_size=(1, 1), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size=(1, 1), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same',
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size=(
        3, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=1024, kernel_size=(3, 3),
              activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3),
              activation=lrelu, kernel_regularizer=l2(5e-4)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense((7*7*5), activation='sigmoid'))
    model.add(Reshape((7, 7, 5)))
    # model.add(Reshape(target_shape=(7, 7, 1)))
    return model


# def yolo_loss_v2():


def prepare_data(images: List[Tuple[np.ndarray, np.ndarray]], grid=7, target_size=(448, 448)) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    for image, bbox in images:
        # First split the bounding box into variables
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + (0.5*w)
        y = bbox[1] + (0.5*h)
        # Descide the size of the cells or anchors of in the image
        shape = image.shape
        cell_w = shape[0] / grid
        cell_h = shape[1] / grid
        # Descide the anchor the object belongs to
        cell_x = int(x // cell_w)
        cell_y = int(y // cell_h)
        print(x, y, cell_x, cell_y)
        # Convert the x,y coordinate of the box to an index
        cell_index = int(cell_x + (cell_y * grid))
        # Calculate the relative position of the bounding box to the grid cell
        scaled_x = x / cell_w
        scaled_y = y / cell_h
        # Calculate the size position of the bounding box to the grid cell
        # Scaled size of the object. Might be bigger than 1 if object is bigger than the cell.
        scaled_w = w / cell_w
        scaled_h = h / cell_h
        probability = 1.0
        head = [
            probability,
            # In our case it would be fine that we only train with one classifier.
        ]
        cells = np.zeros((grid, grid, 6))

        # cells = [[0.0] * 5] * 9

        res = np.array([
            1.0,  # Waldo label, if more classes are allocated, here will be more classes
            scaled_x,
            scaled_y,
            scaled_w,
            scaled_h,
            1.0,  # Probability of object in cell,
        ])

        cells[cell_x, cell_y] = res
        yield (cv2.resize(image, target_size), cells)


def iou():
    pass


def yolo_loss_v2(n_classes: int):
    """
    Get the implementation of the loss function for Yolo
    Based on implementation described by Vivek Maskara
    (https://www.maskaravivek.com/post/yolov1/)

    Arguments:
    ----------
        n_classes (int): The number of classes in the dataset

    Returns:
    --------
        (Func[[Tensor, Tensor], float]): The loss function for a given prediction

    """
    @tf.function
    def yolo_loss_v2_impl(y_true, y_pred):
        # Get the classes (in our case 1).
        label_class = y_true[..., 0]
        # Get the bounding boxes.
        print(k.shape(y_true), k.shape(y_pred), n_classes)
        label_box = y_true[..., n_classes-1:n_classes+5-1]

        label_prob = y_true[..., n_classes+5-1]

        # Get the classes from the predictions
        pred_class = y_pred[..., 0]
        # >= 7x7: [ waldo: 0..1 ]

        # Get the bounding boxes from the predictions
        pred_box = y_pred[..., n_classes:n_classes+5-1]
        # >= 7x7: [ x: 0..1, y: 0..1, w: 0.., h 0..]

        loss_x = k.square(pred_box[..., 0] - label_box[..., 0])
        # >= 7x7  x: 0..1
        loss_y = k.square(pred_box[..., 1] - label_box[..., 1])
        # >= 7x7 x: 0..1

        pos_loss = k.sum((loss_x + loss_y) * label_prob)

        # TODO, Loss function crashes when sqrt
        # Probably negative number for sqrt
        loss_w = k.square(pred_box[..., 2] - label_box[..., 2])
        loss_h = k.square(pred_box[..., 3] - label_box[..., 3])

        size_loss = k.sum((loss_w + loss_h) * label_prob)
        classifier_loss = k.sum(pred_class - label_class)

        return pos_loss + size_loss + classifier_loss

    return yolo_loss_v2_impl


LR: List[Tuple[int, float]] = [
    (0, 0.01),
    (10, 0.001),
    (20, 0.0001)

]


@LearningRateScheduler
def learning_rate(epoch: int, lr: float) -> float:
    for e, l in reversed(LR):
        if e <= epoch:
            return l


if __name__ == "__main__":
    images = load_images(get_waldos("256"))
    images = list(images)
    # print(images)
    images = list(prepare_data(images))
    training = images[:int(len(images)//2)]
    end = int(((len(images)//2)+(len(images)//2))*0.2)
    validation = images[int(len(images)//2):end]
    images_input = [i[0] for i in training]
    bounding_boxes = [i[1] for i in training]

    v_img = [i[0] for i in training]
    v_bb = [i[1] for i in training]
    testing = images[int(len(images)//2): -1]
    test_img = [i[0] for i in testing]
    t_img = [i[0] for i in training]
    t_bb = [i[1] for i in training]

    tds = tf.data.Dataset.from_tensor_slices(
        (t_img, t_bb)).batch(1)
    # print(bounding_boxes)
    # print("BBox", bounding_boxes[0].shape)
    ds = tf.data.Dataset.from_tensor_slices(
        (images_input, bounding_boxes)).batch(1)

    vds = tf.data.Dataset.from_tensor_slices(
        (v_img, v_bb)).batch(1)
    model = get_yolo_model()
    model.summary()

    try:
        i = 5
        loss = tf.py_function(
            yolo_loss, [model.input, model.output], Tout=tf.float64)
        model.compile(optimizer=Adam(),
                      loss=yolo_loss_v2(1))
        res = model.fit(ds, epochs=40, validation_data=vds,
                        callbacks=[learning_rate])
        model.save("yolo_waldo" + str(i), overwrite=True)
        with open(f"training{i}", "w") as f:
            f.write(str(res))
        p = model.predict(tds)
        print(p)

        with open(f"test{i}", "w"):
            f.write(dumps(list(p)))

    except Exception as e:
        print(e)
