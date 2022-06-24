from waldo import get_waldos, load_images, argument_dataset
from util_rcnn import calculate_intersection_over_union
from typing import List, Tuple
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from random import shuffle
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


def get_inceptionresnet_v2():
    return InceptionResNetV2(include_top="true", weights="coco")


def get_vgg16_model(pre_trained: bool = True) -> Model:
    if pre_trained:
        model = VGG16(weights='imagenet', include_top=True)
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
        Conv2D(64, (3, 3), name="conv32_1", activation="relu"),
        Conv2D(64, (3, 3), name="conv32_2", activation="relu"),
        MaxPool2D((2, 2)),
        Conv2D(128, (3, 3), name="con64_1", activation="relu"),
        Conv2D(128, (1, 1), name="con64_2", activation="relu"),

        Conv2D(128, (3, 3), name="con64_3", activation="relu"),
        MaxPool2D((2, 2)),
        Conv2D(128, (3, 3), name="conv64_4", activation="relu"),
        Conv2D(128, (3, 3), name="conv64_5", activation="relu"),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid")
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
        shuffle(regions)
        img_true = []
        lbl_true = []
        img_false = []
        lbl_false = []
        for region in regions:

            # if true_count >= 60 and false_count >= 60:
            # break
            bb_region = np.array([
                region[0], region[1],
                region[0] + region[2], region[1] + region[3]
            ])
            # print(region)

            iou_score = calculate_intersection_over_union(lbl, bb_region)
            iou_scores.append(iou_score)
            if iou_score > 0.6:
                # true_count += 1
                # train_labels.append([1., 0.])
                lbl_true.append([1.])
                current = get_region(
                    img, (region[0], region[1]), (region[2], region[3], False))
                print(True)
                img_true.append(current)
                # train_images.append(cv2.flip(current, 0))

            if iou_score < 0.1 and false_count < 60:
                false_count += 1
                lbl_false.append([0.])
                img_false.append(get_region(
                    img, (region[0], region[1]), (region[2], region[3]), False))
                print(False)
        total = min(len(img_false), len(img_true))

        print("Total", total)
        for i in img_true:
            cv2.imshow("True img", i)
            cv2.waitKey(100)

        train_images.extend(img_true[:total])
        train_images.extend(img_false[:total])
        train_labels.extend(lbl_true[:total])
        train_labels.extend(lbl_false[:total])
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
    model = get_vgg16_model()
    input = model.layers[-2].output
    input = model.output
    # dropout = Dropout
    predictions = Dense(1, activation="softmax")(input)
    return Model(inputs=model.input, outputs=predictions)


def get_region(image, area: List[int], size: List[int], show=False):
    x, y, w, h = area[0], area[1], size[0], size[1]
    img = image[y:y+h, x:x+w].copy()
    if show:
        cv2.imshow(f"Region", img)
        cv2.waitKey(50)
    return img


def predict(image: np.ndarray, model: Model, min_probability=0.7) -> List[np.ndarray]:
    print("Predict with minimal probability", min_probability)
    waldos = []
    regions = list(propose_regions(image, None))

    image_regions = [cv2.resize(
        get_region(image, region[0:2], region[2:4]), (224, 224), interpolation=cv2.INTER_AREA) for region in regions]

    prediction = []
    n = 500
    for i, r in enumerate([image_regions[i:i + n] for i in range(0, len(image_regions), n)]):
        print(i, "/", len(image_regions) / n)
        ds = tf.data.Dataset.from_tensor_slices((r)).batch(1)

        prediction.extend(model.predict(ds))
    # prediction.extend(p)

    for i in range(len(image_regions)):
        # result = model.predict(new_img)
        if prediction[i][0] > min_probability:
            waldos.append(regions[i])
    return waldos


def get_augmentation_model():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            # layers.RandomRotation(0.2),
            layers.RandomContrast(factor=0.2),
            tf.keras.layers.RandomBrightness(factor=0.2)
        ]
    )


def train(model: Model = get_custom_model(), save: str = "rcnn") -> Model:
    waldos = list(load_images(get_waldos("256")))
    training_waldos = waldos[:int(round((len(waldos) * 0.5)))]
    verification_waldos = waldos[int(
        round(len(waldos) * 0.5)): int(round(len(waldos) * 0.8))]
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
        (img, lbl)).batch(2).shuffle(buffer_size=2000).map(lambda x, y: (augmentation(x), y))
    vds = tf.data.Dataset.from_tensor_slices(
        (vimg, vlbl)).batch(2)

    model.compile(Adam(0.000001), loss="bce",
                  metrics=['accuracy'])
    model.fit(ds, epochs=15, validation_data=vds)
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
                    img, (bb_region[0], bb_region[1]), (bb_region[2], bb_region[3]), True)
                cv2.imshow("Region", cut)
                cv2.waitKey(0)


if __name__ == "__main__":
    train()
