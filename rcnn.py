from keras.layers import Conv2d, Dense, MaxPool2D, Flatten, Pool
from keras.models import Model, Sequential


def get_vgg16_model() -> Model:
    return Sequential([
        Conv2d(64, (3, 3), name="conv64_1"),
        Conv2d(64, (3, 3), name="conv64_2"),
        MaxPool2D((2, 2)),
        Conv2d(128, (3, 3), name="conv128_1"),
        Conv2d(128, (3, 3), name="conv128_2"),
        MaxPool2D((2, 2)),
        Conv2d(256, (3, 3), name="conv256_1_1"),
        Conv2d(256, (3, 3), name="conv256_1_2"),
        MaxPool2D((2, 2)),
        Conv2d(256, (3, 3), name="conv256_2_1"),
        Conv2d(256, (3, 3), name="conv256_2_2"),
        Flatten(),
        Dense(4096),
        Dense(4096),
        Dense(4096)
    ])


def prepare_data():
    pass
