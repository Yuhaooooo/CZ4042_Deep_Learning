from other.utils.utils import *

dir_path = os.path.dirname(os.path.realpath(__file__))

x_train = np.load(os.path.join(dir_path, 'other', 'npy', 'x_train_char.npy'), allow_pickle=True)
x_test = np.load(os.path.join(dir_path, 'other', 'npy', 'x_test_char.npy'), allow_pickle=True)
y_train = np.load(os.path.join(dir_path, 'other', 'npy', 'y_train_char.npy'), allow_pickle=True)
y_test = np.load(os.path.join(dir_path, 'other', 'npy', 'y_test_char.npy'), allow_pickle=True)

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 50

num_layers=2
lr = 0.01


def rnn_model(x, withDropout):

    word_vectors = tf.reshape(
      tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256])

    word_list = tf.unstack(word_vectors, axis=1)

    def rnn_cell():
        return tf.nn.rnn_cell.BasicRNNCell(num_units=HIDDEN_SIZE, activation=tf.nn.softsign)

    stacked_rnn = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_layers)])

    encoding, _ = tf.nn.dynamic_rnn(cell=stacked_rnn,
                                inputs=word_list,
                                dtype=tf.float32)


    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
    if withDropout:
        logits = tf.layers.dropout(logits)

    return word_vectors, word_list, encoding, logits


def train(withDropout):

    global x_train, x_test, y_train, y_test, no_epochs

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    word_vectors, word_list, encoding, logits = rnn_model(x, withDropout)

    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('word_vectors: ', sess.run([tf.shape(word_vectors)], {x: x_train, y_: y_train}))
    print('word_list: ', sess.run([tf.shape(word_list)], {x: x_train, y_: y_train}))
    print('encoding: ', sess.run([tf.shape(encoding)], {x: x_train, y_: y_train}))
    print('logits: ', sess.run([tf.shape(logits)], {x: x_train, y_: y_train}))


    entropy_on_training = []
    accuracy_on_testing = []

    timeRecoder = TimeRecoder()
    timeRecoder.start()

    for e in range(no_epochs):

        epoch_loss = []
        x_train, y_train = shuffle(x_train, y_train)

        # training
        for i in range(len(y_train)//batch_size):
            _, loss_  = sess.run([train_op, entropy], {x: x_train[i*batch_size: (i+1)*batch_size], y_: y_train[i*batch_size: (i+1)*batch_size]})
            epoch_loss.append(loss_)
        
        entropy_on_training.append(sum(epoch_loss)/len(epoch_loss))
        
        # testing
        predict = sess.run([logits], {x: x_test})
        accuracy_on_testing.append(accuracy_score(list(y_test), list(np.argmax(np.array(predict[0]), axis=1))))
        
        
        print('epoch %d: entropy: %f, accuracy: %f' % (e, entropy_on_training[-1], accuracy_on_testing[-1]))
        
    timeRecoder.end()

    if withDropout:

        np.save(os.path.join(dir_path, 'other', 'npy', 'char_2rnn_entropy_on_training_withDropout.npy'), np.array(entropy_on_training))
        np.save(os.path.join(dir_path, 'other', 'npy', 'char_2rnn_accuracy_on_testing_withDropout.npy'), np.array(accuracy_on_testing))

        #plot
        plt.figure()
        plt.plot(entropy_on_training)
        plt.plot(accuracy_on_testing)
        plt.title('entropy / accuracy')
        plt.xlabel('epoch')
        plt.legend(['entropy_on_training', 'accuracy_on_testing',], loc='upper left')
        plt.savefig(os.path.join(dir_path, 'other', 'figure', 'char_2rnn_withDropout.png'))  

    else:
        np.save(os.path.join(dir_path, 'other', 'npy', 'char_2rnn_entropy_on_training_withoutDropout.npy'), np.array(entropy_on_training))
        np.save(os.path.join(dir_path, 'other', 'npy', 'char_2rnn_accuracy_on_testing_withoutDropout.npy'), np.array(accuracy_on_testing))

        #plot
        plt.figure()
        plt.plot(entropy_on_training)
        plt.plot(accuracy_on_testing)
        plt.title('entropy / accuracy')
        plt.xlabel('epoch')
        plt.legend(['entropy_on_training', 'accuracy_on_testing',], loc='upper left')
        plt.savefig(os.path.join(dir_path, 'other', 'figure', 'char_2rnn_withoutDropout.png'))  







def main():
    print('\n\n {} \n Without Dropout ... \n {} \n\n'.format('-'*40, '-'*40,))
    train(withDropout=False)

    tf.reset_default_graph()

    print('\n\n {} \n With Dropout ... \n {} \n\n'.format('-'*40, '-'*40,))
    train(withDropout=True)


if __name__ == '__main__':
    main()