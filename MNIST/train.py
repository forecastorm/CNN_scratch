import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import time
import random
import gzip
import cv2
from convnet import *


def extract_data(filename, num_images, IMAGE_WIDTH):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH * IMAGE_WIDTH)
        return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def view_image(image):
    cv2.imshow("window", image)
    cv2.waitKey(0)


def initialize_theta(NUM_OUTPUT, l_in):
    return 0.01 * np.random.rand(NUM_OUTPUT, l_in)


def get_shape(np_array):
    print(np_array.shape)


if __name__ == '__main__':
    train_images_num = 6000
    valid_images_num = 1000
    image_width = 28
    img_depth = 1

    # shape by (6000, 784)
    train_images = extract_data('train-images-idx3-ubyte.gz', train_images_num, 28)
    # shape by (6000,1)
    train_labels = extract_labels('t10k-labels-idx1-ubyte.gz', train_images_num).reshape(train_images_num,1)

    # shape by (1000, 784)
    valid_images = extract_data("t10k-images-idx3-ubyte.gz", valid_images_num, 28)
    # shape by (1000,1)
    valid_labels = extract_labels('t10k-labels-idx1-ubyte.gz', valid_images_num).reshape(valid_images_num,1)

    # this just brings down the data to -0.5 to 0.5
    train_images -= int(np.mean(train_images))
    train_images /= int(np.std(train_images))

    valid_images -= int(np.mean(train_images))
    valid_images /= int(np.std(train_images))
    train_data = np.hstack((train_images, train_labels))

    # Initializing all the parameters
    filt1 = {}
    filt2 = {}
    bias1 = {}
    bias2 = {}
    filt1_num = 8
    filt2_num = 8
    output_num = 10
    filter_size = 3
    learning_rate = 0.01
    # number of samples used in one iteration
    batch_size = 20
    # number of times the algorithm goes through the entire training samples
    epochs_num = 2

    # gradient momentum
    MU = 0.95

    # # each filter1[i] is shaped by (1,3,3)
    for i in range(filt1_num):
        filt1[i] = initialise_param_lecun_normal(filter_size, 1, scale=1.0, distribution='normal')
        bias1[i] = 0

    for i in range(filt2_num):
        filt2[i] = initialise_param_lecun_normal(filter_size, 1, scale=1.0, distribution='normal')
        bias2[i] = 0

    # w1: 26 --- filter1 output width
    w1 = image_width - filter_size + 1
    # w2: 24 ---- filter2 output width
    w2 = w1 - filter_size + 1
    # shaped by (10,1152) --- (output_num, 12 * 12 * 8 )
    theta3 = initialize_theta(output_num, (w2 // 2) * (w2 // 2) * filt2_num)

    # bias3 shaped by (10,1)
    bias3 = np.zeros((output_num, 1))
    # error, loss
    cost = []
    acc = []

    print("Learning Rate: " + str(learning_rate))
    print("Batch Size: " + str(batch_size))

    for epoch in range(epochs_num):
        # so the order probably matters
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0,train_images_num, batch_size)]
        x = 0
        # each batch is shaped by (20,785)
        # 20 samples, in a row, 784 cols for image, 1 col for label
        for batch in batches:
            start_time = time.time()
            # bias1, bias2 has 8 length, each filter has a bias
            out = momentumGradDescent(batch,learning_rate,image_width,img_depth,MU,filt1,filt2,bias1,bias2,theta3,bias3,cost,acc)
            [filt1, filt2, bias1, bias2, theta3, bias3, cost, acc] = out
            epoch_acc = round(np.sum(acc[epoch * train_images_num / batch_size:]) / (x + 1), 2)
            per = float(x + 1) / len(batches) * 100
            print("Epoch:" + str(round(per, 2)) + "% Of " + str(epoch + 1) + "/" + str(epochs_num) + ", Cost:" + str(
                cost[-1]) + ", B.Acc:" + str(acc[-1] * 100) + ", E.Acc:" + str(epoch_acc))