

from cgi import test
from email.generator import Generator
from logging import root
import shutil
from typing import List, Tuple, NamedTuple
from skimage import io
import numpy as np
from os import walk
from pathlib import Path
from random import randint, randrange
from collections import namedtuple
from random import shuffle
from shutil import copy
import xmltodict
# from cv2 import imread
import cv2
from typing import Iterable
from scipy.ndimage import shift, zoom
from random import random


root_path = Path(__file__).parent.joinpath("wheres-waldo/Hey-Waldo")


class ImageSet(NamedTuple):
    waldo: List[Path]
    not_waldo: List[Path]


class TrainingSet(NamedTuple):
    train_set: List[Tuple[str, Path]]
    test_set: List[Tuple[str, Path]]
    validation_set: List[Tuple[str, Path]]


def get_images(subset: str) -> ImageSet:
    subset_path = root_path.joinpath(subset)
    notwaldo = subset_path.joinpath("notwaldo")
    waldo = subset_path.joinpath("waldo")
    return ImageSet(waldo=list(waldo.iterdir()), not_waldo=list(notwaldo.iterdir()))


def get_waldos(folder: str) -> List[Tuple[Path, np.ndarray]]:
    dir = root_path.joinpath(folder).joinpath("waldo")
    files = sorted(list(set([f.stem for f in dir.iterdir()])))
    results = []
    for f in files:
        try:
            image = dir.joinpath(f + ".jpg")
            xml_document = xmltodict.parse(
                dir.joinpath(f + ".xml").read_text())
            if "object" not in xml_document["annotation"] or "bndbox" not in xml_document["annotation"]["object"]:
                continue

            xml = xml_document["annotation"]["object"]["bndbox"]
            arr = np.array([
                float(xml["xmin"]), float(xml["ymin"]), float(
                    xml["xmax"]), float(xml["ymax"])
            ])
            results.append((image, arr))
        except Exception as e:
            print(xml_document)
            raise e
    return results


def get_not_waldo(folder: str):
    dir = root_path.joinpath(folder).joinpath("notwaldo")
    files = sorted(list(set([f.stem for f in dir.iterdir()])))
    for f in files:
        image = str(dir.joinpath(f + ".jpg"))
        yield cv2.imread(image)


def load_images(images: List[Tuple[Path, np.ndarray]]):
    for path, bbox in images:
        img = cv2.imread(str(path))
        yield (img, bbox)


def change_brightness(img: np.ndarray, mult: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.minimum(np.maximum(
        hsv * np.array([1., 1., mult], dtype=np.float32), 255), 0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def change_saturation(img: np.ndarray, mult: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.minimum(np.maximum(
        hsv * np.array([1., mult, 1.], dtype=np.float32), 255), 0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flip_v(img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flipped_img = cv2.flip(img, 0)
    shape = img.shape
    width = shape[0]
    new_lbl = lbl.copy()
    new_lbl[0] = width - lbl[2]
    new_lbl[2] = width - lbl[0]

    print(f"Flipped vertical with width {width}", lbl, new_lbl)
    return flipped_img, new_lbl


def stretch_w(img: np.ndarray, lbl: np.ndarray, factor: float = None):
    if factor is None:
        factor = random() + 0.5
    w = img.shape[0]
    h = img.shape[1]
    zoomed = zoom(img, (factor, 1., 1.))
    return crop(zoomed, lbl, np.array([0., 0., img.shape[0], img.shape[1]]))


def stretch_h(img: np.ndarray, lbl: np.ndarray, factor: float = None):
    if factor is None:
        factor = random() + 0.5
    w = img.shape[0]
    h = img.shape[1]
    zoomed = zoom(img, (1., factor, 1.))
    return crop(zoomed, lbl, np.array([0., 0., img.shape[0], img.shape[1]]))


def flip_h(img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flipped_img = cv2.flip(img, 1)
    shape = img.shape
    height = shape[1]
    new_lbl = lbl.copy()
    new_lbl[1] = height - lbl[3]
    new_lbl[3] = height - lbl[1]

    print(f"Flipped horizontal with height {height}", lbl, new_lbl)
    return flipped_img, new_lbl


def shift_image(image, lbl,  x_amount=None, y_amount=None) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = image.shape
    if x_amount is None:
        x_amount = int(round(x * (random() - 0.5) * .5))
    if y_amount is None:
        y_amount = int(round(y * (random() - 0.5) * .5))
    shifted = shift(image, (x_amount, y_amount, 0))

    return shifted, np.minimum(np.maximum(lbl + np.array([x_amount, y_amount, x_amount, y_amount]), 0.), np.array([x, y, x, y]))


def argument_dataset(images: List[Tuple[np.ndarray, np.ndarray]]):
    print(f"Argumenting {len(images)} images...")
    images = list(images)
    result = [*images]
    result.extend([flip_v(img, lbl) for img, lbl in images])
    result.extend([flip_h(img, lbl) for img, lbl in images])
    result.extend([(img2, lbl2) for img2, lbl2, v in (
        stretch_w(img, lbl) for img, lbl in result) if v])
    result.extend([(img2, lbl2) for img2, lbl2, v in (
        stretch_h(img, lbl) for img, lbl in result) if v])
    result.extend([(img2, lbl2) for img2, lbl2, v in (
        cut_waldo(img, lbl) for img, lbl in images) if v])
    result.extend([(img2, lbl2) for img2, lbl2, v in (
        cut_waldo(img, lbl, 0) for img, lbl in images) if v])
    # result.extend([stretch_h(img, lbl) for img, lbl in result])
    result.extend([shift_image(img, lbl) for img, lbl in images])
    result.extend([(change_brightness(img, random() + 0.5), lbl)
                  for img, lbl, in images])
    result.extend([(change_saturation(img, random() + 0.5), lbl)
                  for img, lbl, in images])

    shuffle(result)
    print("Argumenting finished!")
    return result


def crop(image: np.ndarray, label: np.ndarray, region: np.ndarray):
    print("Region", region)
    x1, y1, x2, y2 = region.round().astype(int).tolist()
    cropped_image = image[x1:x2, y1: y2]
    shifted_label = label - np.array([x1, y1, x1, y1])
    gt0 = np.maximum(shifted_label, 0.)
    lt_max = np.minimum(gt0, np.array([x2 - x1, y2-y1, x2 - x1, y2-y1]))
    assert (lt_max >= 0.).all()
    assert (lt_max <= np.array([x2 - x1, y2-y1, x2 - x1, y2-y1])).all()
    valid = lt_max[0] == lt_max[2] or lt_max[1] == lt_max[3]

    return cropped_image, lt_max, not valid


def stretch_to(image: np.ndarray, label: np.ndarray, new_size: np.ndarray):
    w, h, c = image.shape
    factor = new_size / np.array([w, h])
    new_label = label * np.array(factor[0], factor[1], factor[0], factor[1])
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, new_label


def cut_waldo(image: np.ndarray, label: np.ndarray, padding: int = None):
    w, h, c = image.shape
    if padding is None:
        padding = int(round(random() * (min([w, h]))))
    area = label + \
        np.array([max(-padding, 0), max(-padding, 0),
                 min(padding, w), min(padding, h)])
    return crop(image, label, area)


def sample_images(image_set: ImageSet, training=0.7, validiation=0.2, testing=0.1) -> TrainingSet:

    waldos = [("waldo", img)for img in image_set.waldo]
    notwaldos = [("not_waldo", img)for img in image_set.not_waldo]
    len_waldos = min(len(waldos), len(notwaldos))
    len_notwaldos = min(len(waldos), len(notwaldos))

    shuffle(waldos)
    shuffle(notwaldos)

    train_set = [
        *[waldos.pop() for i in range(int(round(len_waldos * training)))],
        *[notwaldos.pop()
          for i in range(int(round(len_notwaldos * training)))],
    ]
    shuffle(train_set)
    validate_set = [
        *[waldos.pop() for i in range(int(round(len_waldos * testing)))],
        *[notwaldos.pop() for i in range(int(round(len_notwaldos * testing)))],
    ]
    shuffle(validate_set)
    test_set = [
        *[waldos.pop() for i in range(int(round(len_waldos * testing)))],
        *[notwaldos.pop() for i in range(int(round(len_notwaldos * testing)))],
    ]
    shuffle(test_set)
    return TrainingSet(train_set=train_set, test_set=test_set, validation_set=validate_set)


# def copy_trainingset(trainingset: TrainingSet, destination: Path):

    # return ((waldo[waldo_indices]))
if __name__ == "__main__":
    print(get_waldos("64"))
