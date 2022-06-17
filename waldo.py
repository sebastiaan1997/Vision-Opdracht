

from cgi import test
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
    files = set([f.stem for f in dir.iterdir()])
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


def load_images(images: List[Tuple[Path, np.ndarray]]):
    for path, bbox in images:
        yield (cv2.imread(str(path)), bbox)


def change_brightness(img: np.ndarray, mult: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.maximum(np.multiply(
        hsv[:, :, 2].astype(np.float64), mult).astype(np.uint8), np.ones_like(hsv) * 255, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def argument_dataset(images: List[Tuple[np.ndarray, np.ndarray]]):
    return shuffle([
        *[change_brightness(img, 0.5) for img, lbl in images],
        *[change_brightness(img, 1.5) for img, lbl in images]
        * images
    ])


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
