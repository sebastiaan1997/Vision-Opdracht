from waldo import get_waldos, load_images
from util_rcnn import calculate_intersection_over_union
from typing import List, Tuple
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.applications.vgg16 import VGG16
import numpy as np
from typing import Generator
import keras
import cv2
from random import choice
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Input
from keras.models import Model, Sequential

# from tensorflow.data import Dataset


# https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55


ss = None


def propose_regions(image: np.ndarray, amount=2000):
    global ss
    if ss is None:
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return list(ss.process())[:amount]


def get_vgg16_model(pre_trained: bool = True) -> Model:
    if pre_trained:
        model = VGG16(weights='imagenet', include_top=True)
        for layer in (model.layers)[:15]:
            layer.trainable = False
        return model

    return Sequential([
        Conv2D(64, (3, 3), name="conv64_1"),
        Conv2D(64, (3, 3), name="conv64_2"),
        MaxPool2D((2, 2)),
        Conv2D(128, (3, 3), name="conv128_1"),
        Conv2D(128, (3, 3), name="conv128_2"),
        MaxPool2D((2, 2)),
        Conv2D(256, (3, 3), name="conv256_1_1"),
        Conv2D(256, (3, 3), name="conv256_1_2"),
        MaxPool2D((2, 2)),
        Conv2D(256, (3, 3), name="conv256_2_1"),
        Conv2D(256, (3, 3), name="conv256_2_2"),
        Flatten(),
        Dense(4096),
        Dense(4096),
        Dense(4096)
    ])


def get_custom_model():
    return Sequential([
        Conv2D(64, (3, 3), name="conv64_1", activation="relu"),
        Conv2D(64, (3, 3), name="conv64_2", activation="relu"),
        MaxPool2D((2, 2)),
        Conv2D(128, (3, 3), name="conv128_1", activation="relu"),
        Conv2D(128, (3, 3), name="conv128_2", activation="relu"),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(128),
        Dense(64),
        Dense(16),
        Dense(2, activation="softmax")
    ])


def generate_testdataset(dataset: List[Tuple[np.ndarray, np.ndarray]]):
    train_images = []
    train_labels = []
    for img, lbl in dataset:
        regions = propose_regions(img, 100)
        for region in regions[:1000]:
            iou_score = calculate_intersection_over_union(lbl, region)
            train_labels.append(iou_score > 0.7 and 1. or 0.)
            train_images.append(get_region(img, region[0], region[1]))
    return train_images, train_labels


def prepare_images_from_data(data: List[Tuple[np.ndarray, np.ndarray]], target_size: Tuple[int, int]) -> List[np.ndarray]:
    regions = []
    for image, label in data:
        x, y, w, h = label[0], label[1], label[2] - \
            label[0], label[3] - label[1]
        region = get_region(image, [x, y], [w, h])
        regions.append(cv2.resize(region, target_size,
                       interpolation=cv2.INTER_AREA))
    return regions


def get_model() -> Model:
    model = get_custom_model()
    return model
    # input = model.layers[-2].output
    input = model.output
    predictions = Dense(2, activation="softmax")(input)
    return Model(inputs=model.input, outputs=predictions)


def get_region(image, area: List[int], size: List[int]):
    x, y, w, h = area[0], area[1], size[0], size[1]
    return image[y:y+h, x:x+w].copy()


def predict(image: np.ndarray) -> List[np.ndarray]:
    model = get_model()
    waldos = []
    for region in propose_regions(image):
        region = get_region(region)
        x, y, w, h = region
        region_image = get_region(image, [x, y], [w, h])
        resized_image = cv2.resize(
            region_image, (224, 224), interpolation=cv2.INTER_AREA)
        new_img = np.expand_dims(resized_image, axis=0)
        result = model.predict(new_img)
        if result[0][0] > 0.7:
            waldos.append(region)
    return waldos


def train(model: Model, save: str) -> Model:
    waldos = list(load_images(get_waldos("256")))
    img, lbl = generate_testdataset(waldos)
    model.fit(img, lbl)
    model.save(save, True)


def test_cutout():
    waldos = get_waldos("256")
    img = cv2.imread(str(waldos[0][0]))
    cv2.imshow("Uncut", img)
    cv2.waitKey(0)
    regions = propose_regions(img, 2000)
    print(regions)
    while True:
        reg = choice(regions)
        cut = get_region(img, (reg[0], reg[1]), (reg[2], reg[3]))
        cv2.imshow("Region", cut)
        cv2.waitKey(0)


if __name__ == "__main__":
    test_cutout()
