from model_lib import get_augmentation_model
from waldo import get_waldos, load_images
import cv2
import itertools as itr
from typing import Tuple, List, Iterable
from math import sqrt
from pathlib import Path
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical, image_dataset_from_directory, plot_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Rescaling, Layer, Reshape, LeakyReLU, Input, LocallyConnected2D
from keras.optimizers import Adam

from keras.models import Sequential, Model

from keras.regularizers import l2
import keras.backend as k

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import keras.backend as K
import pickle
from json import dumps
import appel as a
from tensorflow.keras.regularizers import l2
from keras import layers
import keras


def get_augmentation_model():
    return keras.Sequential(
        [
            layers.RandomContrast(factor=.5),
            tf.keras.layers.RandomBrightness(factor=.5)
        ]
    )


yolo_leaky_activation = "relu"


def get_yolo_model(img_h=448, img_w=448, load_model=None) -> Model:
    """Implementatie van het YoLo CNN zoals beschreven in paper"""
    lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

    model = Sequential([
        Input((448, 448, 3)),
        Conv2D(64, (7, 7), padding="same", strides=(2, 2),
               activation=lrelu, kernel_regularizer=l2(5e-4)),
        MaxPooling2D((2, 2), strides=((2, 2)), padding="same"
                     ),
        Conv2D(192, (3, 3), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        MaxPooling2D((2, 2), strides=((2, 2)), padding="same"),
        Conv2D(128, (1, 1), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(256, (3, 3), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(256, (1, 1), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(512, (3, 3), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        MaxPooling2D((2, 2), strides=((2, 2)), padding="same"),
        Conv2D(256, (1, 1), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(512, (3, 3), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(256, (1, 1), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(512, (3, 3), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(256, (1, 1), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(512, (3, 3), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(256, (1, 1), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(512, (3, 3), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(512, (1, 1), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(1024, (3, 3), padding="same",
               activation=lrelu, kernel_regularizer=l2(5e-4)),
        MaxPooling2D((2, 2), strides=((2, 2)), padding="same"),

        Conv2D(512, (1, 1), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(1024, (3, 3), padding="same",
               activation=lrelu, kernel_regularizer=l2(5e-4)),

        Conv2D(512, (1, 1), padding="same", activation=lrelu,
               kernel_regularizer=l2(5e-4)),
        Conv2D(1024, (3, 3), padding="same",
               activation=lrelu, kernel_regularizer=l2(5e-4)),
        Conv2D(1024, (3, 3), padding="same",
               activation=lrelu, kernel_regularizer=l2(5e-4)),
        Conv2D(1024, (3, 3), padding="same", strides=(2, 2),
               activation=lrelu, kernel_regularizer=l2(5e-4)),
        Conv2D(1024, (3, 3), padding="same",
               activation=lrelu, kernel_regularizer=l2(5e-4)),
        Conv2D(1024, (3, 3), padding="same",
               activation=lrelu, kernel_regularizer=l2(5e-4)),
        Flatten(),
        Dense(512, kernel_regularizer=l2(5e-4)),
        Dense(1024, kernel_regularizer=l2(5e-4)),

        Dropout(0.5),
        Dense(7*7*6, kernel_regularizer=l2(5e-4)),
        Reshape((7, 7, 6))
    ])

    # if load_model is not None:
    # model.load_weights()
    return model


# def yolo_loss_v2():


def prepare_data(images: List[Tuple[np.ndarray, np.ndarray]], grid=7, target_size=(448, 448)) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    for index, (image, bbox) in enumerate(images):
        # First split the bounding box into variables
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        x = float(bbox[0]) + (0.5*float(w))
        y = float(bbox[1]) + (0.5*float(h))
        # Descide the size of the cells or anchors of in the image
        try:
            shape = image.shape

        except Exception as e:
            print(f"failed at index {index}")
            raise e
        cell_w = shape[1] / grid
        cell_h = shape[0] / grid
        # Descide the anchor the object belongs to
        cell_x = int(x // cell_w)
        cell_y = int(y // cell_h)
        print(x, y, cell_x, cell_y)
        # Convert the x,y coordinate of the box to an index
        # Calculate the relative position of the bounding box to the grid cell

        scaled_x = (float(x) / cell_w) % cell_w
        scaled_y = (float(y) / cell_h) % cell_h
        # Calculate the size position of the bounding box to the grid cell
        # Scaled size of the object. Might be bigger than 1 if object is bigger than the cell.
        scaled_w = w / float(shape[1])
        scaled_h = h / float(shape[0])
        # Create the grid
        cells = np.zeros((grid, grid, 6))

        # Create the cell
        res = np.array([
            1.0,  # Waldo label, if more classes are allocated, here will be more classes
            scaled_x,
            scaled_y,
            scaled_w,
            scaled_h,
            1.0,  # Probability of object in cell,
        ])

        cells[cell_x, cell_y] = res
        yield (np.divide(np.array(cv2.resize(image, target_size), np.float64), 255.), cells)


def yolo_loss(image_size, grid_size=7):
    """
    Get the Yolo loss function

    """
    grid_factor = np.array(image_size) / grid_size

    def yolo_loss_impl(y_true, y_pred):
        """
        Implementation of the Yolo Loss function for VISN-2020
        """
        # Get the shape of the prediction
        pred_shape = tf.shape(y_pred)
        # Set the base number of loss
        loss = 0.
        for p_i in range(pred_shape[0]):
            # Get the prediction
            prediction = y_pred[p_i]
            # Get the ground_truth
            truth = y_true[p_i]
            # Initialize all individiual loss components
            pos_loss = 0.
            size_loss = 0.
            classification_loss = 0.
            confidence_loss = 0.
            # Loop over all prediction boxes
            for x in range(tf.shape(prediction)[0]):
                for y in range(tf.shape(prediction)[1]):
                    # Get the predicted x and y
                    conf_pred = prediction[x, y, 0]
                    x_pos_pred = float(
                        x) + prediction[x, y, 1] * grid_factor[0]
                    y_pos_pred = float(
                        y) + prediction[x, y, 2] * grid_factor[1]
                    w_pred = prediction[x, y, 3] * image_size[0]
                    h_pred = prediction[x, y, 4] * image_size[1]
                    c_pred = prediction[x, y, 5]

                    # Convert the predicted values for iou
                    # Translate middle point to top-left and bottom right
                    x_min_pred = x_pos_pred - (w_pred * .5)
                    x_max_pred = x_pos_pred + (w_pred * .5)

                    y_min_pred = y_pos_pred - (h_pred * .5)
                    y_max_pred = y_pos_pred + (h_pred * .5)

                    # Prepare for IOU challange
                    best_iou = 0.
                    best_x = -1
                    best_y = -1
                    # Loop over all ground truths
                    for x_t in range(prediction.shape[0]):
                        for y_t in range(prediction.shape[1]):
                            # Get the ground truth x and y
                            x_pos_true = (float(
                                x_t) + truth[x_t, y_t, 1]) * grid_factor[0]
                            y_pos_true = (float(
                                y_t) + truth[x_t, y_t, 2]) * grid_factor[1]
                            w_true = truth[x_t, y_t, 3] * image_size[0]
                            h_true = truth[x_t, y_t, 4] * image_size[1]
                            c_true = truth[x_t, y_t, 5]
                            conf_true = truth[x_t, y_t, 0]

                            x_min_true = x_pos_true - (w_true * .5)
                            x_max_true = x_pos_true + (w_true * .5)
                            y_min_true = y_pos_true - (h_true * .5)
                            y_max_true = y_pos_true + (h_true * .5)

                            x_start = tf.maximum(x_min_true, x_min_pred)
                            y_start = tf.maximum(y_min_true, y_min_pred)

                            x_end = tf.minimum(x_max_pred, x_max_true)
                            y_end = tf.minimum(y_max_pred, y_max_true)
                            # If there is no intersection, skip
                            if tf.logical_or(x_end < x_start, y_end < y_start):
                                continue

                            # Calculate intersection
                            intersection = (x_end - x_start) * \
                                (y_end - y_start)

                            # Calculate areas of both bounding boxes
                            lhs_area = h_true * w_true
                            rhs_area = h_pred * w_pred
                            # return intersection / ((lhs_area + rhs_area) - intersection)

                            iou = intersection / \
                                ((lhs_area + rhs_area) - intersection)
                            # If the iou value is bigger than 6, apply the normal size and loss
                            if tf.greater(iou, best_iou):
                                best_iou = iou
                                best_x = x_t
                                best_y = y_t
                    for x_t in range(prediction.shape[0]):
                        for y_t in range(prediction.shape[1]):
                            conf_true = truth[x, y, 0]
                            if x_t == best_x and y_t == best_y:
                                x_pos_true = (float(
                                    x_t) + truth[x_t, y_t, 1]) * grid_factor[0]
                                y_pos_true = (float(
                                    y_t) + truth[x_t, y_t, 2]) * grid_factor[1]
                                w_true = truth[x_t, y_t, 3] * image_size[0]
                                h_true = truth[x_t, y_t, 4] * image_size[1]
                                c_true = truth[x_t, y_t, 5]
                                # Add the positional loss  to the pos_loss
                                pos_loss += (x_pos_true - x_pos_pred) ** 2
                                pos_loss += ((y_pos_true - y_pos_pred) ** 2)
                                # Add the positional size to the size_loss
                                size_loss += ((w_true - w_pred) ** 2)
                                size_loss += ((h_true - h_pred) ** 2)
                                classification_loss += (c_true - c_pred) ** 2
                                confidence_loss += (conf_true - conf_pred) ** 2
                            else:
                                confidence_loss += .5 * \
                                    ((conf_true - conf_pred) ** 2)

            loss += pos_loss + size_loss + classification_loss + confidence_loss
        return loss
    return yolo_loss_impl


tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="tensorboard_log", histogram_freq=1)


def get_bounding_boxes(prediction: np.ndarray, target_image_size: Tuple[int, int], grid_size=7, min_prob=.7) -> Iterable[Tuple[int, int, int, int]]:
    shape = prediction.shape
    for box_x in range(shape[0]):
        for box_y in range(shape[1]):
            box = prediction[box_x, box_y]
            print(box)
            probability = box[0]
            label = box[5]

            print("Prob", probability, "Label", label)
            if probability > min_prob:
                t_size = np.array(target_image_size)
                cell_mult = t_size.astype(float) / 7.

                x = box_x + box[1] * cell_mult[0]
                y = box_y + box[2] * cell_mult[1]
                w, h = box[3:5] * t_size
                x1 = x - (w*.5)
                x2 = x + (w*.5)

                y1 = y - (h*.5)
                y2 = y + (h*.5)

                yield (np.minimum(np.maximum(np.array([x1, y1, x2, y2]), 0.), np.array([*t_size, *t_size])) / float(grid_size)) * np.array(target_image_size, np.float32).repeat(2)


def predict(model: Model, image: np.ndarray):
    prediction = model.predict(np.array([image]))
    return get_bounding_boxes(prediction, image.shape)





@ LearningRateScheduler
def learning_rate(epoch: int, lr: float) -> float:
    """ Schedules the learning rate for YOLO based on the LR"""
    LR: List[Tuple[int, float]] = [
        (0, 1e-2),
        (60, 1e-3),
        (95, 1e-4),
        # (190, 1e-6)
        # (120, 0.1e-6)
    ]
    for e, l in reversed(LR):
        if e <= epoch:
            return l


if __name__ == "__main__":
    model_checkpoint = ModelCheckpoint(
        "best_yolo_model_loss", monitor="loss", save_best_only=True, mode="min", period=5)

    model_checkpoint2 = ModelCheckpoint(
        "best_yolo_model_valloss", monitor="val_loss", save_best_only=True, mode="min", period=5)

    # if __name__ == "__main__":
    # images = argument_dataset(load_images(get_waldos("256")))
    images = list(load_images(get_waldos("256")))

    images = list(prepare_data(images))
    # print("Images", len(images))
    # # print(images)

    # images = [(cv2.resize(img, (448, 448)) / 255., lbl)
    #           for img, lbl in a.appel_to_yolo_notation(a.load_appels(True))]
    train_amount = int(len(images)*.75)

    training = images[:train_amount]
    end = int(((len(images)*.75)+(len(images)*80))*0.2)
    validation = images[train_amount:]
    # print(training)
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
        (images_input, bounding_boxes)).batch(1).shuffle(buffer_size=500)
    augmentation = get_augmentation_model()
    # ds = ds.map(lambda x, y: (augmentation(x), y))
    # ds.batch(5)

    vds = tf.data.Dataset.from_tensor_slices(
        (v_img, v_bb)).batch(1)
    model = get_yolo_model()
    model.build((-1, 224, 224, 3))
    model.summary()

    try:
        i = 5
        loss = tf.py_function(
            yolo_loss, [model.input, model.output], Tout=tf.float64)
        adm = Adam(learning_rate=0.5e-4, beta_1=0.9,
                   beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adm,
                      loss=yolo_loss((448, 448)), metrics=["accuracy"])
        history = model.fit(ds, epochs=100, validation_data=vds, batch_size=1,
                            callbacks=[])
        with open('/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        model.save("yolo_10_segmentation", overwrite=True)
        p = model.predict(tds)
        print(p)

        with open(f"test{i}", "w") as f:
            f.write(dumps(list(p)))

    except Exception as e:
        print(e)
