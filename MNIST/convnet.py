import numpy as np
import gzip
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def initialise_param_lecun_normal(FILTER_SIZE, IMG_DEPTH, scale=1.0, distribution='normal'):
    if scale <= 0.:
        raise ValueError('`scale` must be a positive float. Got:', scale)

    distribution = distribution.lower()
    if distribution not in {'normal'}:
        raise ValueError('Invalid `distribution` argument: '
                         'expected one of {"normal", "uniform"} '
                         'but got', distribution)

    scale = scale
    distribution = distribution
    fan_in = FILTER_SIZE * FILTER_SIZE * IMG_DEPTH
    scale = scale
    stddev = scale * np.sqrt(1. / fan_in)
    shape = (IMG_DEPTH, FILTER_SIZE, FILTER_SIZE)
    return np.random.normal(loc=0, scale=stddev, size=shape)


def initialize_theta(NUM_OUTPUT, l_in):
    return 0.01 * np.random.rand(NUM_OUTPUT, l_in)


def maxpool(filter, pool_matrix_size, stride):
    filter_num, filter_width, filter_width = filter.shape
    # pool shape by ( 8, 12, 12 )
    pool = np.zeros(
        (filter_num, (filter_width - pool_matrix_size) // stride + 1, (filter_width - pool_matrix_size) // stride + 1))
    for jj in range(filter_num):
        i = 0
        # for every row
        while i < filter_width:
            j = 0
            # for every col
            while j < filter_width:
                # store max value
                pool[jj, i // 2, j // 2] = np.max(filter[jj, i:i + pool_matrix_size, j:j + pool_matrix_size])
                # move stride step
                j += stride
            i += stride
    return pool


def softmax_cost(out, y):
    eout = np.exp(out, dtype=np.float)  # we dont have 128 a typo fixed
    probs = eout / sum(eout)
    p = sum(y * probs)
    cost = -np.log(p)  ## (Only data loss. No regularised loss)
    return cost, probs


## Returns idexes of maximum value of the array
def nanargmax(a):
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    if np.isnan(a[multi_idx]):
        nan_count = np.sum(np.isnan(a))
        # In numpy < 1.8 use idx = np.argsort(a, axis=None)[-nan_count-1]
        idx = np.argpartition(a, -nan_count - 1, axis=None)[-nan_count - 1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx


def ConvNet(image, label, filt1, filt2, bias1, bias2, theta3, bias3):
    # calculating first convolution layer
    depth, width, width = image.shape
    # filt1_num, filt2_num is 8
    filt1_num = len(filt1)
    filt2_num = len(filt2)
    # f is 3
    _, f, f = filt1[0].shape
    # w1 : 28 - 3 + 1 = 26
    w1 = width - f + 1
    # w2 : 26 - 3 + 1 = 24
    w2 = w1 - f + 1

    # conv1 shape by (8,26,26)
    conv1 = np.zeros((filt1_num, w1, w1))
    # conv2 shape by (8,24,24)
    conv2 = np.zeros((filt2_num, w2, w2))
    # all 8 filt1
    for i in range(filt1_num):
        # all 26 filt1 row
        for j in range(w1):
            # all 26 filt1 col
            for k in range(w1):
                # each image, every row, to row +3, every col, to col +3 , multiply to the current filter
                product = image[:, j:j + f, k:k + f] * filt1[i]
                # sum of product -- dot product
                # bias1 has length of 8,
                conv1[i, j, k] = np.sum(product) + bias1[i]
    # relu activation
    conv1[conv1 <= 0] = 0

    # all 8 filt2
    for i in range(filt2_num):
        # all 24 filt2 row
        for j in range(w2):
            # all 24 filt2 col
            for k in range(w2):
                # each image, every row, to row +3, every col, to col +3 , multiply to the current filter
                product = conv1[:, j:j + f, k:k + f] * filt2[i]
                # sum of product -- dot product
                # bias2 has length of 8,
                conv2[i, j, k] = np.sum(product) + bias2[i]
    # relu activation
    conv2[conv2 <= 0] = 0

    # pooled layer with 2*2 size and stride 2,2
    # pool layer shape (8, 12, 12 )
    pooled_layer = maxpool(conv2, 2, 2)

    # shape by (1152, 1 )
    fully_connect_layer1 = pooled_layer.reshape((w2 // 2) * (w2 // 2) * filt2_num, 1)

    # out shape by (10, 1), bias3 shape by ( 10, 1), theta3 shape by (10,1152)
    out = theta3.dot(fully_connect_layer1) + bias3

    # using soft max function to get cost
    # probs sums up to 1
    # cost shape (1,), probs shape(10,1)
    cost, probs = softmax_cost(out, label)
    # if the index of max value is the same as labels,
    # in other worlds, if the highest probability for prediction is correct
    if np.argmax(out) == np.argmax(label):
        acc = 1
    else:
        acc = 0

    # back propagation to get gradient
    # doubt shape (10,1) --- all differences between prediction and label
    dout = probs - label

    # dtheta3 shape (10, 1152), is calculated by 'differences . T(full_connected_layer) '
    # so differences are updated to each 1152 nodes
    dtheta3 = dout.dot(fully_connect_layer1.T)

    # shape by (10,1)
    dbias3 = sum(dout.T).T.reshape((10, 1))

    # shape by (1152,1)
    dfully_connect_layer1 = theta3.T.dot(dout)
    # shape by (8, 12, 12 )
    dpool = dfully_connect_layer1.reshape(filt2_num, w2 // 2, w2 // 2)

    # shape by (8,24,24)
    dconv2 = np.zeros((filt2_num, w2, w2))

    for jj in range(filt2_num):
        i = 0
        # for every filter2 output row, w2 = 24
        while (i < w2):
            j = 0
            # for every filter2 output col
            # max pool the max value in conv2, put dpool to dconv2
            while (j < w2):
                (a, b) = nanargmax(conv2[jj, i:i + 2, j:j + 2])
                dconv2[jj, i + a, j + b] = dpool[jj, i // 2, j // 2]
                j += 2
            i += 2
    dconv2[conv2 <= 0] = 0
    # shape by (8,26,26)
    dconv1 = np.zeros((depth, w1, w1))

    dfilt1 = {}
    dfilt2 = {}
    dbias1 = {}
    dbias2 = {}

    for i in range(filt1_num):
        dfilt1[i] = np.zeros((depth, f, f))
        dbias1[i] = 0
    for i in range(filt2_num):
        dfilt2[i] = np.zeros((depth, f, f))
        dbias2[i] = 0

    for i in range(filt2_num):
        # each row
        for j in range(w2):
            # each col
            for k in range(w2):
                # update dfilt2,
                # so all 8 conv1 filters, each times moving 3 by 3,
                # is multiplied to dconv2 to produce new dfilt2
                print("conv1 shape")
                print((dconv2[i, j, k] * conv1[:, j:j + f, k:k + f]).shape)
                dfilt2[i] += dconv2[i, j, k] * conv1[i, j:j + f, k:k + f]
                # dconv1 is updated to dconv2 * filt2
                dconv1[:, j:j + f, k:k + f] += dconv2[i, j, k] * filt2[i]
        dbias2[i] = np.sum(dconv2[i])
    dconv1[dconv1 <= 0] = 0

    for i in range(filt1_num):
        for j in range(w1):
            for k in range(w1):
                dfilt1[i] += dconv1[i, j, k] * image[:, j:j + f, k:k + f]

        dbias1[i] = np.sum(dconv1[i])

    return [dfilt1, dfilt2, dbias1, dbias2, dtheta3, dbias3, cost, acc]


def momentumGradDescent(batch, LEARNING_RATE, image_width, image_depth, MU, filt1, filt2, bias1, bias2, theta3, bias3,
                        cost, acc):
    # shaped by (20, 784)
    train_images = batch[:, 0:-1]
    # shaped by (20, 1, 28, 28)
    train_images = train_images.reshape(len(batch), image_depth, image_width, image_width)
    # shaped by (20,)
    train_labels = batch[:, -1]

    num_correct = 0
    cost_ = 0
    # batch_size is 20
    batch_size = len(batch)
    dfilt1 = {}
    dfilt2 = {}
    dbias1 = {}
    dbias2 = {}
    v1 = {}
    v2 = {}
    bv1 = {}
    bv2 = {}

    for k in range(len(filt1)):
        dfilt1[k] = np.zeros(filt1[0].shape)
        dbias1[k] = 0
        v1[k] = np.zeros(filt1[0].shape)
        bv1[k] = 0
    for k in range(len(filt2)):
        dfilt2[k] = np.zeros(filt2[0].shape)
        dbias2[k] = 0
        v2[k] = np.zeros(filt2[0].shape)
        bv2[k] = 0
    # shaped by (10,1152 )
    dtheta3 = np.zeros(theta3.shape)
    # shaped by (10,1)
    dbias3 = np.zeros(bias3.shape)
    # shaped by (10,1152)
    v3 = np.zeros(theta3.shape)
    # shaped by (10,1)
    bv3 = np.zeros(bias3.shape)

    # batch size is 20
    for i in range(batch_size):
        # current_image shape by (1,28,28)
        current_image = train_images[i]
        current_label = np.zeros((theta3.shape[0], 1))
        # current label shape by (10,1)
        # one hot encode the correct label
        current_label[int(train_labels[i]), 0] = 1
        # fetching gradient for the current parameters
        [dfilt1_, dfilt2_, dbias1_, dbias2_, dtheta3_, dbias3_, curr_cost, acc_] = ConvNet(current_image,
                                                                                           current_label, filt1, filt2,
                                                                                           bias1, bias2, theta3, bias3)

        # accumulates for each sample
        for j in range(len(filt2)):
            dfilt2[j] += dfilt2_[j]
            dbias2[j] += dbias2_[j]

        for j in range(len(filt1)):
            dfilt1[j] += dfilt1_[j]
            dbias1[j] += dbias1_[j]

        dtheta3 += dtheta3_
        dbias3 += dbias3_
        cost_ += curr_cost
        num_correct += acc_

    for j in range(len(filt1)):
        # how much to learn? How much to update filt1?
        v1[j] = MU * v1[j] - LEARNING_RATE * dfilt1[j] / batch_size
        filt1[j] += v1[j]
        bv1[j] = MU * bv1[j] - LEARNING_RATE * dbias1[j] / batch_size
        bias1[j] += bv1[j]

    for j in range(0, len(filt2)):
        v2[j] = MU * v2[j] - LEARNING_RATE * dfilt2[j] / batch_size
        filt2[j] += v2[j]
        bv2[j] = MU * bv2[j] - LEARNING_RATE * dbias2[j] / batch_size
        bias2[j] += bv2[j]

    v3 = MU * v3 - LEARNING_RATE * dtheta3 / batch_size
    theta3 += v3
    bv3 = MU * bv3 - LEARNING_RATE * dbias3 / batch_size
    bias3 += bv3

    cost_ = cost_ / batch_size
    cost.append(cost_)
    accuracy = float(n_correct) / batch_size
    acc.append(accuracy)

    return [filt1, filt2, bias1, bias2, theta3, bias3, cost, acc]
