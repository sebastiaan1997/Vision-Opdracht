from ast import arg
from distutils.log import debug
from email.mime import base
from unittest import result
from waldo import get_waldos, load_images, argument_dataset
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
from keras.optimizers import Adam, SGD
from keras.backend import less
from keras.models import Sequential, Model
from re import X
from keras.regularizers import l2
import keras.backend as k
import os
import datetime
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from json import dumps


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

    model = Sequential([
        Input((448, 448, 3)),
        Conv2D(64, (7, 7), padding="same", strides=(2, 2), activation=lrelu),
        MaxPooling2D((2, 2), strides=((2, 2)), padding="same"),
        Conv2D(192, (3, 3), padding="same", activation=lrelu),
        MaxPooling2D((2, 2), strides=((2, 2)), padding="same"),
        Conv2D(128, (1, 1), padding="same", activation=lrelu),
        Conv2D(256, (3, 3), padding="same", activation=lrelu),
        Conv2D(256, (1, 1), padding="same", activation=lrelu),
        Conv2D(512, (3, 3), padding="same", activation=lrelu),
        MaxPooling2D((2, 2), strides=((2, 2)), padding="same"),
        Conv2D(256, (1, 1), padding="same", activation=lrelu),
        Conv2D(512, (3, 3), padding="same", activation=lrelu),
        Conv2D(256, (1, 1), padding="same", activation=lrelu),
        Conv2D(512, (3, 3), padding="same", activation=lrelu),
        Conv2D(256, (1, 1), padding="same", activation=lrelu),
        Conv2D(512, (3, 3), padding="same", activation=lrelu),
        Conv2D(256, (1, 1), padding="same", activation=lrelu),
        Conv2D(512, (3, 3), padding="same", activation=lrelu),
        Conv2D(512, (1, 1), padding="same", activation=lrelu),
        Conv2D(1024, (3, 3), padding="same", activation=lrelu),
        MaxPooling2D((2, 2), strides=((2, 2)), padding="same"),

        Conv2D(512, (1, 1), padding="same", activation=lrelu),
        Conv2D(1024, (3, 3), padding="same", activation=lrelu),

        Conv2D(512, (1, 1), padding="same", activation=lrelu),
        Conv2D(1024, (3, 3), padding="same", activation=lrelu),
        Conv2D(1024, (3, 3), padding="same", activation=lrelu),
        Conv2D(1024, (3, 3), padding="same", strides=(2, 2), activation=lrelu),
        Conv2D(1024, (3, 3), padding="same", activation=lrelu),
        Conv2D(1024, (3, 3), padding="same", activation=lrelu),
        Flatten(),
        Dense(4092),
        Dense(7*7*6),
        Reshape((7, 7, 6))
    ])
    return model


# def yolo_loss_v2():


def prepare_data(images: List[Tuple[np.ndarray, np.ndarray]], grid=7, target_size=(448, 448)) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    for index, (image, bbox) in enumerate(images):
        # First split the bounding box into variables
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + (0.5*w)
        y = bbox[1] + (0.5*h)
        # Descide the size of the cells or anchors of in the image
        try:
            shape = image.shape

        except Exception as e:
            print(f"failed at index {index}")
            raise e
        cell_w = shape[0] / grid
        cell_h = shape[1] / grid
        # Descide the anchor the object belongs to
        cell_x = int(x // cell_w)
        cell_y = int(y // cell_h)
        print(x, y, cell_x, cell_y)
        # Convert the x,y coordinate of the box to an index
        cell_index = int(cell_x + (cell_y * grid))
        # Calculate the relative position of the bounding box to the grid cell

        scaled_x = (x - (cell_w * cell_x)) / cell_w
        scaled_y = (y - (cell_h * cell_y)) / cell_h
        # Calculate the size position of the bounding box to the grid cell
        # Scaled size of the object. Might be bigger than 1 if object is bigger than the cell.
        scaled_w = w / cell_w
        scaled_h = h / cell_h
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
        try:
            cells[cell_x, cell_y] = res
            yield (np.divide(np.array(cv2.resize(image, target_size), np.float64), 255.), cells)
        except:
            continue


def get_bounding_box(prediction: np.ndarray, grid_size: int) -> np.ndarray:
    probabilities = 1 / (1 + np.exp(prediction[..., 0]))
    best_bet_x, best_bet_y = np.unravel_index(
        np.argmax(probabilities)(grid_size, grid_size))

    coordinates = prediction[best_bet_x,  best_bet_y[best_bet_x], 1:5]
    coordinates[2] = np.exp(coordinates[2])
    coordinates[3] = np.exp(coordinates[3])
    absolute_coords = coordinates + \
        np.array([best_bet_x, best_bet_y[best_bet_x], 0, 0])
    return absolute_coords / float(grid_size)


def yolo_loss_v2(n_classes: int, debug_print: bool = False, result_print: bool = True):
    """
    Get the implementation of the loss function for Yolo
    Based on implementation described by Vivek Maskara
    (https://www.maskaravivek.com/post/yolov1/)
    And the implementation descibed from GitHub author "experiencor"
    (https://github.com/experiencor/keras-yolo2/blob/master/Yolo%20Step-by-Step.ipynb)

    Arguments:
    ----------
        n_classes (int): The number of classes in the dataset
        debug_print (bool): Enables the debug prints

    Returns:
    --------
        (Func[[Tensor, Tensor], float]): The loss function for a given prediction

    """
    @tf.function
    def yolo_loss_v2_impl(y_true, y_pred):
        """
        """
        # Get the classes (in our case 1 waldo).
        label_class = y_true[..., 5]
        # Get the bounding boxes.
        label_box = y_true[..., 1:5]

        label_prob = y_true[..., 0]

        print(y_true)
        mask_shape = (7, 7)
        # coord_mask = tf.zeros(mask_shape)
        confidence_mask = tf.zeros(mask_shape)
        classification_mask = tf.zeros(mask_shape)

        # Get the classes from the predictions
        pred_class = y_pred[..., 0]
        # >= 7x7: [ waldo: 0..1 ]

        # Get the bounding boxes from the predictions
        pred_box = y_pred[..., 1:5]
        pred_prob = y_pred[..., 5]

        coord_mask = tf.expand_dims(label_prob, -1)
        # >= 7x7: [ x: 0..1, y: 0..1, w: 0.., h 0..]
        # Get the coordinates and sizes of the predicted labels

        # >= [[x: 0..1, y: 0..1]]
        pred_coord = pred_box[..., 0:2]
        # >= [[ w:0.., y: 0.. ]]

        if debug_print:
            tf.print("Size in", tf.nn.relu(pred_box[..., 2:4]))
        # >= Not mentioned in paper, but in real implementation
        pred_size = tf.exp(pred_box[..., 2:4])
        pred_half_size = pred_size * 0.5  # >= [[ hw: 0.5*w, hh: 0.5*h]]

        lbl_coord = label_box[..., 0:2]  # >= [[x: 0..1, y: 0..1]]
        lbl_size = label_box[..., 2:4]   # >= [[ w:0.., y: 0.. ]]
        lbl_half_size = lbl_size * 0.5   # >= [[ hw: 0.5*w, hh: 0.5*h]]

        min_lbl_coord = lbl_coord - lbl_half_size
        max_lbl_coord = lbl_coord + lbl_half_size

        min_pred_coord = pred_coord - pred_half_size
        max_pred_coord = pred_coord + pred_half_size

        intersect_mins = k.maximum(min_lbl_coord, min_pred_coord)
        intersect_max = k.minimum(max_lbl_coord, max_pred_coord)

        intersect_box = k.maximum(intersect_max - intersect_mins, 0.)
        intersect_area = intersect_box[..., 0] * intersect_box[..., 1]

        lbl_area = lbl_size[..., 0] * lbl_size[..., 1]
        pred_area = pred_size[..., 0] * pred_size[..., 1]

        iou = tf.truediv(intersect_area, lbl_area + pred_area - intersect_area)
        box_confidence = iou * label_prob

        box_class_confidence = box_confidence * label_class

        pred_coord_4 = pred_coord
        pred_size_4 = pred_size

        pred_half_size_4 = pred_size_4 * .5
        pred_coord_min_4 = pred_coord_4 - pred_half_size_4
        pred_coord_max_4 = pred_coord_4 + pred_half_size_4

        intersect_min_4 = tf.maximum(pred_coord_min_4, min_lbl_coord)
        intersect_max_4 = tf.minimum(pred_coord_max_4, max_lbl_coord)
        intersect_size_4 = tf.maximum(intersect_max_4 - intersect_min_4, 0.)
        intersect_area_4 = intersect_size_4[..., 0] * intersect_size_4[..., 1]

        pred_area_4 = pred_size_4[..., 0] * pred_size_4[..., 1]
        iou_4 = tf.truediv(intersect_area_4, pred_area_4 +
                           lbl_area - intersect_area_4)
        best_iou = tf.reduce_max(iou_4, axis=2)

        confidence_mask = confidence_mask + \
            tf.cast(best_iou < .6, tf.float32) * \
            (1 - label_prob)
        confidence_mask = confidence_mask + label_prob

        class_mask = label_prob + label_class

        no_boxes_mask = tf.cast(coord_mask < .5, tf.float32)

        nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, tf.float32))
        nb_confidence_box = tf.reduce_sum(
            tf.cast(confidence_mask > 0.0, tf.float32))
        nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, tf.float32))
        print(coord_mask, lbl_coord, lbl_coord)
        loss_pos = tf.reduce_sum(
            tf.square(pred_coord - lbl_coord) * coord_mask)

        if debug_print:
            tf.print("Loss for size", "Lbl_size", lbl_size, "Pred size", pred_size,
                     tf.square(lbl_size - pred_size))
        # Should be sqrt, but looking for option to limit values
        pred_size_gt_0 = tf.maximum(pred_size, 0.)
        lbl_size_gt_0 = tf.maximum(lbl_size, 0.)
        # tf.print("pred_size_gt_0", pred_size_gt_0)
        # tf.print("lbl_size_gt_0", lbl_size_gt_0)
        individual_loss = tf.square(
            ((tf.sqrt(pred_size_gt_0)) - (tf.sqrt(lbl_size_gt_0))) * coord_mask)
        # tf.print("individual_loss", individual_loss)
        loss_size = tf.reduce_sum(individual_loss)
        loss_confidence = tf.reduce_sum(
            tf.square(label_prob-pred_prob) * confidence_mask)
        # loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=label_class, logits=pred_class)
        pred_class = pred_class
        loss_class = tf.reduce_sum(
            tf.square(label_class - pred_class) * class_mask)
        if result_print or debug_print:
            tf.print("\n losses:", loss_pos, loss_size,
                     loss_confidence, loss_class, "\n")
        loss = loss_pos + loss_size + \
            loss_confidence + loss_class
        return loss

    return yolo_loss_v2_impl


tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="tensorboard_log", histogram_freq=1)
LR: List[Tuple[int, float]] = [
    (0, 1e-4),
    (90, 1e-5)
    # (120, 0.1e-6)
]


def get_bounding_boxes(prediction: np.ndarray, target_image_size: Tuple[int, int], grid_size=7) -> Iterable[Tuple[int, int, int, int]]:
    shape = prediction.shape
    for box_x in range(shape[0]):
        for box_y in range(shape[1]):
            box = prediction[box_x, box_y]
            probability = box[0]
            label = box[5]

            print("Prob", probability, "Label", label)
            if probability > 0.5:
                x = box[1]
                y = box[2]

                w = np.exp(box[3]) / 2.
                h = np.exp(box[4]) / 2.

                x1 = float(box_x) + x - w
                x2 = float(box_x) + x + w

                y1 = float(box_y) + y - h
                y2 = float(box_y) + y + h
                yield (np.array([x1, y1, x2, y2]) / float(grid_size)) * np.array(target_image_size, np.float32).repeat(2)


def predict(model: Model, image: np.ndarray):
    prediction = model.predict(np.array([image]))
    return get_bounding_boxes(prediction, image.shape)


@ LearningRateScheduler
def learning_rate(epoch: int, lr: float) -> float:
    """ Schedules the learning rate for YOLO based on the LR"""
    for e, l in reversed(LR):
        if e <= epoch:
            return l


model_checkpoint = ModelCheckpoint(
    "best_yolo_model", monitor="loss", save_best_only=True, mode="min", period=1)


if __name__ == "__main__":
    images = argument_dataset(load_images(get_waldos("256")))
    # images = load_images(get_waldos("256"))

    images = list(images)
    print("Images", len(images))
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
    model.build((-1, 224, 224, 3))
    model.summary()

    try:
        i = 5
        loss = tf.py_function(
            yolo_loss, [model.input, model.output], Tout=tf.float64)
        model.compile(optimizer=Adam(learning_rate=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0005),
                      loss=yolo_loss_v2(1, False, False))
        res = model.fit(ds, epochs=135, validation_data=vds, batch_size=5,
                        callbacks=[learning_rate, model_checkpoint])
        model.save("yolo_waldo" + str(i), overwrite=True)
        with open(f"training{i}", "w") as f:
            f.write(str(res))
        p = model.predict(tds)
        print(p)

        with open(f"test{i}", "w"):
            f.write(dumps(list(p)))

    except Exception as e:
        print(e)
