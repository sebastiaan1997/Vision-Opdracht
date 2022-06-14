import numpy as np
import waldo as w
from skimage.io import imread






if __name__ == "__main__":
    # clf = svm.SVC(gamma=0.001, C=100)
    clf = MLPClassifier()
    ds = w.get_images("64")
    samples = w.sample_images(ds)

    labels = [
        *(["waldo"] * len(samples.train_set.waldo)),
        *(["not_waldo"] * len(samples.train_set.not_waldo))
    ]
    images = [
        *[np.array(imread(str(p.absolute()), as_gray=True)).flatten()
          for p in samples.train_set.waldo],
        * [np.array(imread(str(p.absolute()), as_gray=True)).flatten() for p in samples.train_set.not_waldo
           ]
    ]
    # print(np.array(imread(samples.train_set.waldo[0], True)).flatten())

    clf.fit(images, labels)

    test_images = [
        *[np.array(imread(str(p.absolute()), as_gray=True)).flatten()
          for p in samples.test_set.waldo],
        * [np.array(imread(str(p.absolute()), as_gray=True)).flatten() for p in samples.test_set.not_waldo
           ]
    ]

    print(clf.predict(test_images))

    # print(labels)
