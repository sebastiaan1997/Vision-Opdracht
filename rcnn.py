from waldo import get_waldos, load_images, argument_dataset
from util_rcnn import calculate_intersection_over_union
from typing import List, Tuple
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import numpy as np
from typing import Generator
import keras
import waldo as w
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
    if amount is None:
        return list(ss.process())
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
    iou_scores = []
    for i, (img, lbl) in enumerate(dataset):
        true_images = []
        false_images = []
        true_labels = []
        false_labels = []
        regions = propose_regions(img, None)
        for region in regions[:1000]:
            bb_region = np.array([
                region[0], region[1],
                region[0] + region[2], region[1] + region[3]
            ])
            # print(region)
            iou_score = calculate_intersection_over_union(lbl, bb_region)

            iou_scores.append(iou_score)
            if iou_score > 0.5 and len(true_images) < 30:
                true_labels.append([1., 0.])
                true_images.append(get_region(
                    img, (region[0], region[1]), (region[2], region[3])))
            if iou_score < 0.3 and len(false_images) < 30:
                false_labels.append([0., 1.])
                false_images.append(get_region(
                    img, (region[0], region[1]), (region[2], region[3])))
        train_images.extend(true_images)
        train_images.extend(false_images)
        train_labels.extend(true_labels)
        train_labels.extend(false_labels)

    print(max(iou_scores))
    return train_images, train_labels


def prepare_images_from_data(data: List[Tuple[np.ndarray, np.ndarray]], target_size: Tuple[int, int]) -> List[np.ndarray]:
    regions = []
    for image, label in data:
        x, y, w, h = label[0], label[1], label[2] - \
            label[0], label[3] - label[1]

        print(label)

        region = get_region(image, [x, y], [w, h])
        regions.append(cv2.resize(region, target_size,
                       interpolation=cv2.INTER_AREA))
    return regions


def get_model() -> Model:
    # model = get_custom_model()
    # return model
    # input = model.layers[-2].output
    model = get_vgg16_model()
    input = model.output
    predictions = Dense(2, activation="softmax")(input)
    return Model(inputs=model.input, outputs=predictions)


def get_region(image, area: List[int], size: List[int]):
    x, y, w, h = area[0], area[1], size[0], size[1]
    return image[y:y+h, x:x+w].copy()


def predict(image: np.ndarray, model: Model) -> List[np.ndarray]:
    waldos = []
    regions = list(propose_regions(image))
    image_regions = [cv2.resize(
        get_region(image, region[0:2], region[2:4]), (224, 224), interpolation=cv2.INTER_AREA) for region in regions]
    data = tf.data.Dataset.from_tensor_slices((image_regions)).batch(1)

    prediction = model.predict(data)

    for i in range(len(image_regions)):
        # result = model.predict(new_img)
        if prediction[i][0] > 0.7:
            waldos.append(regions[i])
    return waldos


def train(model: Model = get_custom_model(), save: str = "rcnn") -> Model:
    waldos = list(load_images(get_waldos("256")))
    training_waldos = waldos[:int(round((len(waldos) * 0.5)))]
    verification_waldos = waldos[int(
        round(len(waldos) * 0.5)): int(round(len(waldos) * 0.7))]
    img, lbl = generate_testdataset(argument_dataset(training_waldos))
    vimg, vlbl = generate_testdataset(argument_dataset(verification_waldos))

    # training_waldos = waldos[:(len(waldos) * 0.5)]

    img = list(map(prepare_image, img))
    ds = tf.data.Dataset.from_tensor_slices((img, lbl)).batch(1)

    vds = tf.data.Dataset.from_tensor_slices((vimg, vlbl)).batch(1)
    model.compile(Adam(0.0001), loss=categorical_crossentropy)
    model.fit(ds, epochs=20, batch_size=20, validation_data=vds)
    model.save(save, True)


def prepare_image(image: cv2.Mat):
    return (cv2.resize(image, ((224, 224))) / 225.)


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
    train()
