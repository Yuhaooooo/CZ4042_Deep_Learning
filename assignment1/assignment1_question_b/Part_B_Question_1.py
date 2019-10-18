import numpy as np
import pandas as pd
import os

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras.regularizers import l2

# plot
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    X_train, X_test, y_train, y_test = np.load(os.path.join('others', 'npy', 'X_train.npy')), np.load(os.path.join('others', 'npy', 'X_test.npy')), np.load(os.path.join('others', 'npy', 'y_train.npy')), np.load(os.path.join('others', 'npy', 'y_test.npy'))

    lr = 1e-3
    decay = 1e-3
    batch_size = 8
    epochs = 10000
    epochs_interval =100
    early_stop_threshold = 1e-2

    model = Sequential([
        Dense(10, input_dim=X_train.shape[1], activation='relu', kernel_initializer='lecun_normal', kernel_regularizer=l2(decay)),
        Dense(y_train.shape[1], kernel_initializer='he_normal', kernel_regularizer=l2(decay))
        ])

    model.compile(optimizer=optimizers.SGD(lr=lr),
                    loss='mse')

    train_loss=[100]
    val_loss=[100]
    early_stop=0

    print('\n\n', '-'*10, '  Question 1(a)  ', '-'*10, '\n\n')

    for i in range(epochs//epochs_interval):
        h = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs_interval, verbose=0, validation_data=(X_test, y_test), shuffle=True)
        train_loss.extend(h.history['loss'])
        val_loss.extend(h.history['val_loss'])
        print('epoch: {}, mse: {}'.format(epochs_interval*(i+1),val_loss[-1]))

    #     if val loss increase less than early stop thrshold, then stop
        if val_loss[-1] * (1+early_stop_threshold) > val_loss[-100] and early_stop==0:
            early_stop = epochs_interval*(i+1)
            print('\n\n', '-'*10, '  Question 1(b)  ', '-'*10, '\n\n')
            print('\n\nnow converge... the early stopping epoch is {}...\n\n'.format(early_stop))


    plt.figure()
    plt.plot(train_loss[1:])
    plt.plot(val_loss[1:])
    plt.title('mse')
    plt.xlabel('epoch')
    plt.legend(['train_mean_squared_error', 'test_mean_squared_error',], loc='upper left')
    plt.savefig(os.path.join('.', 'others', 'plot', 'q1', '10000epoch.png'))

    plt.figure()
    plt.plot(train_loss[1:early_stop])
    plt.plot(val_loss[1:early_stop])
    plt.title('mse')
    plt.xlabel('epoch')
    plt.legend(['train_mean_squared_error', 'test_mean_squared_error',], loc='upper left')
    plt.savefig(os.path.join('.', 'others', 'plot', 'q1', 'earlystop.png'))


    print('\n\n', '-'*10, '  Question 1(c)  ', '-'*10, '\n\n')

    _, X_test_50, _, y_test_50 = train_test_split(X_test, y_test, test_size=50, random_state=42)

    y_pred_50 = model.predict(x=X_test_50)
    print('mean root square error: ', np.sqrt(mean_squared_error(y_test_50, y_pred_50)))

    plt.figure()
    plt.plot(y_test_50, 'ro')
    plt.plot(y_pred_50, 'b+')
    plt.legend(['True Value', 'Pred Value',], loc='upper left')
    plt.savefig(os.path.join('.', 'others', 'plot', 'q1', 'random50mse.png'))


if __name__ == '__main__':
    main()