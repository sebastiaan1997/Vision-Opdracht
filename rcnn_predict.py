import json
from keras.models import load_model, Model
import tensorflow as tf
from waldo import get_waldos, load_images
import cv2
import numpy as np
from rcnn import predict
from typing import List

waldos = list(get_waldos("256"))
raw_dataset = list(load_images(waldos))


dataset = (
    [i[0] for i in raw_dataset]
)

# ds = tf.data.Dataset.from_tensor_slices(dataset).batch(1)

m: Model = load_model("rcnn")


def add_bounding_boxes(image, bboxes: List[np.ndarray]):
    target_image = image.copy()

    for box in bboxes:
        cv2.rectangle(target_image, box[0:2].round().astype(int),
                      box[2:4].round().astype(int), (0, 255, 0), 1)
    return target_image


for i in range(len(dataset)):
    res = predict(dataset[i], m, .6)
    with open(f"rcnn{i}.txt", "w") as f:
        f.write(json.dumps([r.tolist() for r in res]))


bounding_boxes = predict(dataset[index], m, prob*.1)


while True:
    index = int(input("Photo: "))

    prob = float(input("Prob: "))
    bounding_boxes = predict(dataset[index], m, prob*.1)

    img = add_bounding_boxes(dataset[index], bounding_boxes)
    # print(bounding_boxes)
    cv2.imshow("Prediction rcnn", img)
    cv2.waitKey(0)
