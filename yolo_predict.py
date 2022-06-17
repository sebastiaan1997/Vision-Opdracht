from keras.models import load_model, Model
import tensorflow as tf
from yolo import yolo_loss_v2, prepare_data
from waldo import get_waldos, load_images
import cv2
img = list(prepare_data(load_images(get_waldos("256"))))
ds = tf.data.Dataset.from_tensor_slices(
    ([img[25][0].reshape(1, 448, 448, 3)], [img[25][1]]))
m: Model = load_model("yolo_waldo5", custom_objects={
                      "yolo_loss_v2_impl": yolo_loss_v2(1)})
print(m.predict(ds))
print(img[25][1])
