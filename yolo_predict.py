from keras.models import load_model, Model
import tensorflow as tf
from yolo import yolo_loss_v2, prepare_data, get_bounding_box, get_bounding_boxes
from waldo import get_waldos, load_images
import cv2
import numpy as np

waldos = list(get_waldos("256"))
raw_dataset = list(load_images(waldos))


img = list(prepare_data(raw_dataset))

dataset = (
    [i[0] for i in img]
)

ds = tf.data.Dataset.from_tensor_slices(dataset).batch(1)
m: Model = load_model("yolo_waldo5", custom_objects={
                      "yolo_loss_v2_impl": yolo_loss_v2(1)})

predictions = m.predict(ds)


img_size = raw_dataset[0][0].shape
bboxes = [list(get_bounding_boxes(predictions[0], (256, 256), 7))
          for i in range(30)]
print(bboxes[0])


while True:
    index = int(input("Index"))
    print("Show index", index)
    current_prediction = predictions[index]
    current_image = raw_dataset[index][0].copy()
    for bbox in bboxes[index]:
        min_pos = bbox[0:2]
        max_pos = bbox[2:4]
        print("Draw", min_pos, max_pos)
        cv2.rectangle(current_image, min_pos.round().astype(int),
                      max_pos.round().astype(int), (0, 255, 0), 3)
    cv2.imshow("Prediction", current_image)
    cv2.waitKey(0)
# print(img[25][1])
