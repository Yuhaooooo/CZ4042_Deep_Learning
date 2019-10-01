import math
import tensorflow as tf
import numpy as np

NUM_FEATURES = 21
NUM_CLASSES = 3

learning_rate = 0.01
epochs = 5000
seed = 10
np.random.seed(seed)


def scale(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)


def train(param):
    batch_size = param['batch_size']
    hidden_layer_neuron_num = param['hidden_layer_neuron_num']
    weight_decay_parameter = param['weight_decay_parameter']

    train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter=',')
    allX, allY = train_input[1:, :21], train_input[1:, -1].astype(int)
    allX = scale(allX, np.min(allX, axis=0), np.max(allX, axis=0))

    all_x_length = allX.shape[0]
    all_y_length = allY.shape[0]

    trainX = allX[:int(all_x_length * 0.7)]
    testX = allX[int(all_x_length * 0.7):]

    trainY_temp = allY[:int(all_x_length * 0.7)]
    testY_temp = allY[int(all_x_length * 0.7):]
    trainY = np.zeros((trainY_temp.shape[0], NUM_CLASSES))
    trainY[np.arange(trainY_temp.shape[0]), trainY_temp - 1] = 1  # one hot matrix
    testY = np.zeros((testY_temp.shape[0], NUM_CLASSES))
    testY[np.arange(testY_temp.shape[0]), testY_temp - 1] = 1  # one hot matrix

    # construct the ffn
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    weights_hidden = tf.Variable(
        tf.truncated_normal(
            [NUM_FEATURES, hidden_layer_neuron_num],
            stddev=1.0 / math.sqrt(float(NUM_FEATURES))),
        name='weights')
    biases_hidden = tf.Variable(
        tf.zeros([hidden_layer_neuron_num]),
        name='biases')
    hidden = tf.nn.relu(tf.matmul(x, weights_hidden) + biases_hidden)
    weights_out = tf.Variable(
        tf.truncated_normal(
            [hidden_layer_neuron_num, NUM_CLASSES],
            stddev=1.0 / math.sqrt(float(NUM_FEATURES))),
        name='weights')
    biases_out = tf.Variable(
        tf.zeros([NUM_CLASSES]),
        name='biases')
    logits = tf.matmul(hidden, weights_out) + biases_out
    regularization = tf.nn.l2_loss(weights_out) + tf.nn.l2_loss(weights_hidden)

    y = logits

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
        loss = tf.reduce_mean(cross_entropy + weight_decay_parameter * regularization)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    with tf.name_scope('error'):
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
        
    if param['required'] == 'train accuracy and test accuracy':
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_acc = []
            test_acc = []

            for i in range(1, (epochs + 1)):
                num_of_batch = trainX.shape[0] // batch_size + 1
                for j in range(num_of_batch):
                    first_index = (j - 1) * batch_size
                    last_index = j * batch_size
                    if last_index > len(trainX):
                        last_index = len(trainX)
                    train_op.run(feed_dict={x: trainX[first_index:last_index], y_:
                                            trainY[first_index:last_index]})
                train_acc.append(accuracy.eval(feed_dict={x: trainX, y_: trainY}))
                test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
                if i % 100 == 0:
                    print('iter %d: train accuracy %g' % (i, train_acc[i - 1]))
                    print('iter %d: test accuracy %g' % (i, test_acc[i - 1]))

            return train_acc, test_acc
    if param['required'] == 'cross-validation accuracy':
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            cross_validation_values = []
            fold_size = trainX.shape[0] // 5
            fold_indexes = [0, fold_size, fold_size * 2, fold_size * 3, fold_size * 4, trainX.shape[0]]
            for i in range(0, 5):
                fold_trainX = trainX[0:fold_indexes[i]]
                fold_trainX = np.concatenate((fold_trainX, trainX[fold_indexes[i+1]:fold_indexes[5]]), axis=0)
                fold_testX = trainX[fold_indexes[i]:fold_indexes[i+1]]
                fold_trainY = trainY[0:fold_indexes[i]]
                fold_trainY = np.concatenate((fold_trainY, trainY[fold_indexes[i + 1]:fold_indexes[5]]), axis=0)
                fold_testY = trainY[fold_indexes[i]:fold_indexes[i + 1]]
                num_of_batch = fold_trainX.shape[0] // batch_size + 1
                for i in range(1, (epochs + 1)):
                    for j in range(num_of_batch):
                        first_index = (j - 1) * batch_size
                        last_index = j * batch_size
                        if last_index > len(fold_trainX):
                            last_index = len(fold_trainX)
                        train_op.run(feed_dict={x: fold_trainX[first_index:last_index],
                                                y_: fold_trainY[first_index:last_index]})
                    if i % 100 == 0:
                        print(str(i) + 'epoch')
                cross_validation_values.append(error.eval(feed_dict={x: fold_testX, y_: fold_testY}))
                print(cross_validation_values)
            return sum(cross_validation_values) / len(cross_validation_values)
