from other.utils.utils import *

dir_path = os.path.dirname(os.path.realpath(__file__))

x_train = np.load(os.path.join(dir_path, 'other', 'npy', 'x_train_cnn.npy'), allow_pickle=True)
x_test = np.load(os.path.join(dir_path, 'other', 'npy', 'x_test_cnn.npy'), allow_pickle=True)
y_train = np.load(os.path.join(dir_path, 'other', 'npy', 'y_train_cnn.npy'), allow_pickle=True)
y_test = np.load(os.path.join(dir_path, 'other', 'npy', 'y_test_cnn.npy'), allow_pickle=True)

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
no_epochs = 3000
lr = 0.01


def char_cnn_model(x, withDropout):
  
    input_layer = tf.reshape(
      tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

    with tf.variable_scope('CNN_Layer1'):
        conv1 = tf.layers.conv2d(
            input_layer,
            filters=N_FILTERS,
            kernel_size=FILTER_SHAPE1,
            padding='VALID',
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')
        
    with tf.variable_scope('CNN_Layer2'):
        conv2 = tf.layers.conv2d(
            pool1,
            filters=N_FILTERS,
            kernel_size=FILTER_SHAPE2,
            padding='VALID',
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')

    dim = pool2.get_shape()[1].value * pool2.get_shape()[2].value * pool2.get_shape()[3].value 
    
    with tf.variable_scope('CNN_Flatten'):
        flatten = tf.reshape(pool2, [-1, dim])
    
    with tf.variable_scope('ANN'):
        W1 = tf.Variable(tf.truncated_normal([dim, MAX_LABEL], stddev=1.0/np.sqrt(dim)))
        b1 = tf.Variable(tf.zeros([MAX_LABEL]))
        logits = tf.matmul(flatten, W1) + b1
        if withDropout:
            logits = tf.layers.dropout(logits)

    return input_layer, conv1, pool1, conv2, pool2, flatten, logits


def train(withDropout):
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    inputs, conv1, pool1, conv2, pool2, flatten, logits = char_cnn_model(x, withDropout)

    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('input: ', sess.run([tf.shape(inputs)], {x: x_train, y_: y_train}))
    print('conv1: ', sess.run([tf.shape(conv1)], {x: x_train, y_: y_train}))
    print('pool1: ', sess.run([tf.shape(pool1)], {x: x_train, y_: y_train}))
    print('conv2: ', sess.run([tf.shape(conv2)], {x: x_train, y_: y_train}))
    print('pool2: ', sess.run([tf.shape(pool2)], {x: x_train, y_: y_train}))
    print('flatten: ', sess.run([tf.shape(flatten)], {x: x_train, y_: y_train}))
    print('logits: ', sess.run([tf.shape(logits)], {x: x_train, y_: y_train}))

    entropy_on_training = []
    accuracy_on_testing = []

    timeRecoder = TimeRecoder()
    timeRecoder.start()

    for e in range(no_epochs):
        
        # training
        _, loss_  = sess.run([train_op, entropy], {x: x_train, y_: y_train})
        entropy_on_training.append(loss_)
        
        # testing
        predict = sess.run([logits], {x: x_test})
        accuracy_on_testing.append(accuracy_score(list(y_test), list(np.argmax(np.array(predict[0]), axis=1))))
        
        
        print('epoch %d: entropy: %f, accuracy: %f' % (e, entropy_on_training[-1], accuracy_on_testing[-1]))
        
    timeRecoder.end()

    if withDropout:

        np.save(os.path.join(dir_path, 'other', 'npy', 'entropy_on_training_q1_withDropout.npy'), np.array(entropy_on_training))
        np.save(os.path.join(dir_path, 'other', 'npy', 'accuracy_on_testing_q1_withDropout.npy'), np.array(accuracy_on_testing))

        #plot
        plt.figure()
        plt.plot(entropy_on_training)
        plt.plot(accuracy_on_testing)
        plt.title('entropy / accuracy q1 with dropout')
        plt.xlabel('epoch')
        plt.legend(['entropy_on_training', 'accuracy_on_testing',], loc='upper left')
        plt.savefig(os.path.join(dir_path, 'other', 'figure', 'q1_withDropout.png'))  

    else:
        np.save(os.path.join(dir_path, 'other', 'npy', 'entropy_on_training_q1_withoutDropout.npy'), np.array(entropy_on_training))
        np.save(os.path.join(dir_path, 'other', 'npy', 'accuracy_on_testing_q1_withoutDropout.npy'), np.array(accuracy_on_testing))

        #plot
        plt.figure()
        plt.plot(entropy_on_training)
        plt.plot(accuracy_on_testing)
        plt.title('entropy / accuracy q1 without dropout')
        plt.xlabel('epoch')
        plt.legend(['entropy_on_training', 'accuracy_on_testing',], loc='upper left')
        plt.savefig(os.path.join(dir_path, 'other', 'figure', 'q1_withoutDropout.png'))  



def main():
    print('\n\n {} \n Without Dropout ... \n {} \n\n'.format('-'*40, '-'*40,))
    train(withDropout=False)

    tf.reset_default_graph()

    print('\n\n {} \n With Dropout ... \n {} \n\n'.format('-'*40, '-'*40,))
    train(withDropout=True)


if __name__ == '__main__':
    main()