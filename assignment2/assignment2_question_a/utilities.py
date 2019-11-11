import tensorflow.compat.v1 as tf
import numpy as np
import pickle
import time

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
seed = time.time()
np.random.seed(int(seed))
tf.set_random_seed(int(seed))


def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  # python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels - 1] = 1

    return data, labels_


def cnn(images, conv_1_feature_map_num, conv_2_feature_map_num, required):
    # images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    images = tf.reshape(images, [-1, NUM_CHANNELS, IMG_SIZE, IMG_SIZE])
    images = tf.transpose(images, [0, 2, 3, 1])
    print(images.get_shape())

    # Convolution 1
    w1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, conv_1_feature_map_num], stddev=1.0 / np.sqrt(NUM_CHANNELS * 9 * 9)),
                     name='weights_1')
    b1 = tf.Variable(tf.zeros([conv_1_feature_map_num]), name='biases_1')
    conv_1 = tf.nn.relu(tf.nn.conv2d(images, w1, [1, 1, 1, 1], padding='VALID') + b1)
    print(conv_1.get_shape())
    pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_1')
    print(pool_1.get_shape())

    # Convolution 2
    w2 = tf.Variable(tf.truncated_normal([5, 5, conv_1_feature_map_num, conv_2_feature_map_num], stddev=1.0 / np.sqrt(NUM_CHANNELS * 5 * 5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([conv_2_feature_map_num]), name='biases_2')
    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, w2, [1, 1, 1, 1], padding='VALID') + b2)
    print(conv_2.get_shape())
    pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_2')
    print(pool_2.get_shape())
    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
    pool_2_flat = tf.reshape(pool_2, [-1, dim])

    # Fully connected
    w3 = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0 / np.sqrt(float(dim))), name='weights_3')
    b3 = tf.Variable(tf.zeros([300]), name='biases_3')
    fully_connect = tf.nn.relu(tf.matmul(pool_2_flat, w3) + b3)
    print(fully_connect.get_shape())
    if required == 'accuracies using Dropout':
        keep_prob = tf.placeholder(tf.float32)
        # Softmax
        fully_connect_drop = tf.nn.dropout(fully_connect, keep_prob)
        w4 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0 / np.sqrt(300)), name='weights_4')
        b4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
        logits = tf.matmul(fully_connect_drop, w4) + b4
        return logits, conv_1, conv_2, pool_1, pool_2, keep_prob
    else:
        # Softmax
        w4 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0 / np.sqrt(300)), name='weights_4')
        b4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
        logits = tf.matmul(fully_connect, w4) + b4
        return logits, conv_1, conv_2, pool_1, pool_2


def train(param):
    required = param['required']
    learning_rate = param['learning_rate']
    epochs = param['epochs']
    batch_size = param['batch_size']
    conv_1_feature_map_num = param['conv_1_feature_map_num']
    conv_2_feature_map_num = param['conv_2_feature_map_num']

    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    if required == 'accuracies using Dropout':
        logits, conv_1, conv_2, pool_1, pool_2, keep_prob = cnn(x, conv_1_feature_map_num, conv_2_feature_map_num, required)
    else:
        logits, conv_1, conv_2, pool_1, pool_2 = cnn(x, conv_1_feature_map_num, conv_2_feature_map_num, required)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    if required == 'accuracies using Momentum':
        train_step = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(loss)
    elif required == 'accuracies using RMSProp':
        train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    elif required == 'accuracies using Adam':
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    else:   # GD or GD with dropout
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)
    trainX = (trainX - np.min(trainX, axis=0)) / np.max(trainX, axis=0)
    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_accuracies = []
        training_costs = []
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            num_of_batch = trainX.shape[0] // batch_size + 1
            for i in range(num_of_batch):
                first_index = i * batch_size
                last_index = (i + 1) * batch_size
                if last_index > len(trainX):
                    last_index = len(trainX)
                if required == 'accuracies using Dropout':
                    _, loss_ = sess.run([train_step, loss], {x: trainX[first_index:last_index],
                                                             y_: trainY[first_index:last_index], keep_prob: 0.5})
                else:
                    _, loss_ = sess.run([train_step, loss], {x: trainX[first_index:last_index],
                                                             y_: trainY[first_index:last_index]})
            if required == 'accuracies using Dropout':
                test_accuracy = accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0})
            else:
                test_accuracy = accuracy.eval(feed_dict={x: testX, y_: testY})
            test_accuracies.append(test_accuracy)
            training_costs.append(loss_)
            print('epoch', e + 1, 'entropy', loss_, 'accuracy', test_accuracy)

        if required == 'accuracies with feature maps':
            ind = np.random.randint(low=0, high=2000)
            image_1 = testX[ind, :]
            ind = np.random.randint(low=0, high=2000)
            image_2 = testX[ind, :]
            image_1_conv_1_feature_map = sess.run(conv_1, feed_dict={x: [image_1]})
            image_1_conv_2_feature_map = sess.run(conv_2, feed_dict={x: [image_1]})
            image_2_conv_1_feature_map = sess.run(conv_1, feed_dict={x: [image_2]})
            image_2_conv_2_feature_map = sess.run(conv_2, feed_dict={x: [image_2]})
            image_1_pool_1_feature_map = sess.run(pool_1, feed_dict={x: [image_1]})
            image_1_pool_2_feature_map = sess.run(pool_2, feed_dict={x: [image_1]})
            image_2_pool_1_feature_map = sess.run(pool_1, feed_dict={x: [image_2]})
            image_2_pool_2_feature_map = sess.run(pool_2, feed_dict={x: [image_2]})

            images_to_plot = [image_1, image_2, image_1_conv_1_feature_map, image_1_conv_2_feature_map,
                              image_2_conv_1_feature_map, image_2_conv_2_feature_map, image_1_pool_1_feature_map,
                              image_1_pool_2_feature_map, image_2_pool_1_feature_map, image_2_pool_2_feature_map]
            return test_accuracies, training_costs, images_to_plot

        else:
            return test_accuracies, training_costs

