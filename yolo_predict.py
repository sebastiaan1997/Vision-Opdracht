from keras.models import load_model, Model
import tensorflow as tf
from yolo import yolo_loss_v2, prepare_data, get_bounding_box
from waldo import get_waldos, load_images
import cv2
import numpy as np

waldos = list(get_waldos("256"))
raw_dataset = list(load_images(waldos))


img = list(prepare_data(raw_dataset))
ds = tf.data.Dataset.from_tensor_slices(
    ([img[25][0].reshape(1, 448, 448, 3)], [img[25][1]]))
m: Model = load_model("yolo_waldo5", custom_objects={
                      "yolo_loss_v2_impl": yolo_loss_v2(1)})

predictions = m.predict(ds)
index = 0


img_size = raw_dataset[0][0].shape
bbox = get_bounding_box(
    predictions[index], 7) * np.array([img_size[0], img_size[1], img_size[0], img_size[1]])
bbox = np.maximum(bbox, 0.)

min_pos = bbox[0:2] - (bbox[2:4] * 0.5)
max_pos = bbox[0:2] + (bbox[2:4] * 0.5)
print("Bounding box", min_pos, max_pos)

cv2.rectangle(raw_dataset[index][0], min_pos.round().astype(int),
              max_pos.round().astype(int), (255, 0, 0), 2)
cv2.imshow("Prediction", raw_dataset[index][0])
cv2.waitKey(0)
# print(img[25][1])
