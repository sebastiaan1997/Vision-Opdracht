import numpy as np
from typing import List
import cv2
import json
import waldo as w


def add_bounding_boxes(image, bboxes: List[np.ndarray]):
    target_image = image.copy()

    for box in bboxes:
        cv2.rectangle(target_image, box[0:2].round().astype(int),
                      box[2:4].round().astype(int), (0, 255, 0), 1)
    return target_image


if __name__ == "__main__":
    i = int(input("Index "))
    json_obj = None
    with open(f"rcnn{i}.txt") as f:
        txt = f.read()
        json_obj = json.loads(txt)
    if json_obj is None:
        exit()
    json_obj = map(np.array, json_obj)

    waldos = list(w.load_images(w.get_waldos("256")))
    img = add_bounding_boxes(waldos[i][0], json_obj)
    cv2.imshow("Bounding Box", img)
    cv2.waitKey(0)
