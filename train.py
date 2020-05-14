from mnist import MNIST
import numpy as np
from sklearn import svm, metrics
from mnist.loader import MNIST
from preview_mnist import *
mndata = MNIST('D:\\DOWNLOAD\\number\\mnist')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
print("TRAIN")
TRAINING_SIZE = 10000
train_images = get_images("mnist\\train-images-idx3-ubyte", TRAINING_SIZE)
train_images = np.array(train_images)/255
train_labels = get_labels("D:\\DOWNLOAD\\number\\mnist\\train-labels-idx1-ubyte", TRAINING_SIZE)
clf = svm.SVC('scale')
clf.fit(train_images, train_labels)

TEST_SIZE = 500
test_images = get_images("mnist\\t10k-images-idx3-ubyte", TEST_SIZE)
test_images = np.array(test_images)/255
test_labels = get_labels("mnist\\t10k-labels-idx1-ubyte", TEST_SIZE)

print("PREDICT")
predict = clf.predict(test_images)

print("RESULT")
ac_score = metrics.accuracy_score(test_labels, predict)
cl_report = metrics.classification_report(test_labels, predict)
print("Score = ", ac_score)
print(cl_report)
