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
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Input, Dropout
from keras.models import Model, Sequential
from keras import layers
# from tensorflow.data import Dataset


# https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55


ss = None


def propose_regions(image: np.ndarray, amount=2000, fast: bool = False):
    global ss
    if ss is None:
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    if fast:
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

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
        print(i, "of", len(dataset))
        true_count = 0
        false_count = 0
        regions = propose_regions(img, None)
        for region in regions:
            if true_count >= 60 and false_count >= 60:
                break
            bb_region = np.array([
                region[0], region[1],
                region[0] + region[2], region[1] + region[3]
            ])
            # print(region)
            iou_score = calculate_intersection_over_union(lbl, bb_region)
            iou_scores.append(iou_score)
            if iou_score > 0.6 and true_count < 30:
                true_count += 1
                # train_labels.append([1., 0.])
                train_labels.append([1., 0.])
                current = get_region(
                    img, (region[0], region[1]), (region[2], region[3]))
                train_images.append(current)
                # train_images.append(cv2.flip(current, 0))

            if iou_score < 0.3 and false_count < 30:
                false_count += 1

                train_labels.append([0., 1.])
                train_images.append(get_region(
                    img, (region[0], region[1]), (region[2], region[3])))

    # print(max(iou_scores))
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
    # dropout = Dropout
    predictions = Dense(2, activation="softmax")(input)
    return Model(inputs=model.input, outputs=predictions)


def get_region(image, area: List[int], size: List[int]):
    x, y, w, h = area[0], area[1], size[0], size[1]
    return image[y:y+h, x:x+w].copy()


def predict(image: np.ndarray, model: Model, min_probability=0.7) -> List[np.ndarray]:
    waldos = []
    regions = list(propose_regions(image))

    image_regions = [cv2.resize(
        get_region(image, region[0:2], region[2:4]), (224, 224), interpolation=cv2.INTER_AREA) for region in regions]
    data = tf.data.Dataset.from_tensor_slices((image_regions)).batch(1)

    prediction = model.predict(data)

    for i in range(len(image_regions)):
        # result = model.predict(new_img)
        if prediction[i][0] > min_probability:
            waldos.append(regions[i])
    return waldos


def get_augmentation_model():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomContrast(factor=0.2),
            tf.keras.layers.RandomBrightness(factor=0.2)
        ]
    )


def train(model: Model = get_custom_model(), save: str = "rcnn") -> Model:
    waldos = list(load_images(get_waldos("256")))
    training_waldos = waldos[:int(round((len(waldos) * 0.5)))]
    verification_waldos = waldos[int(
        round(len(waldos) * 0.5)): int(round(len(waldos) * 0.7))]
    img, lbl = generate_testdataset(training_waldos)
    vimg, vlbl = generate_testdataset(verification_waldos)
    print(np.asarray(img).shape)
    print(np.asarray(lbl).shape)
    augmentation = get_augmentation_model()

    print(np.asarray(vimg).shape)
    print(np.asarray(vlbl).shape)

    img = list(map(prepare_image, img))
    vimg = list(map(prepare_image, vimg))
    ds = tf.data.Dataset.from_tensor_slices(
        (img, lbl)).batch(16).map(lambda x, y: (augmentation(x), y)).shuffle(buffer_size=2000)
    vds = tf.data.Dataset.from_tensor_slices(
        (vimg, vlbl)).batch(16)

    model.compile(Adam(0.001), loss=categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(ds, epochs=10, validation_data=vds)
    model.save(save, True)


def prepare_image(image: cv2.Mat):
    return (cv2.resize(image, ((224, 224))) / 225.)


def test_cutout():
    waldos = get_waldos("256")
    for i in range(len(waldos)):
        img = cv2.imread(str(waldos[i][0]))
        print(img.shape)
        bbox = waldos[i][1]
        cv2.imshow("Uncut", img)
        cv2.waitKey(0)
        print(bbox)

        cv2.imshow("Cut", get_region(
            img, (int(bbox[0]), int(bbox[1])), (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))))
        cv2.waitKey(0)
        regions = list(propose_regions(img, None))
        # print(regions)
        for region in regions:

            bb_region = np.array([
                region[0], region[1],
                region[0] + region[2], region[1] + region[3]
            ])
            # print(region)
            iou_score = calculate_intersection_over_union(bbox, bb_region)
            if iou_score > 0.6:

                cut = get_region(
                    img, (bb_region[0], bb_region[1]), (bb_region[2], bb_region[3]))
                cv2.imshow("Region", cut)
                cv2.waitKey(0)


if __name__ == "__main__":
    train()
