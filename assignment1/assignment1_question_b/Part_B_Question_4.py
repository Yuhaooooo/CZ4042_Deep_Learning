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

import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = np.load(os.path.join('others', 'npy', 'X_train2.npy')), np.load(os.path.join('others', 'npy', 'X_test2.npy')), np.load(os.path.join('others', 'npy', 'y_train.npy')), np.load(os.path.join('others', 'npy', 'y_test.npy'))

epochs=20000
epochs_interval =100
batch_size=8
early_stop_threshold = 1e-2
lr=1e-3
decay = 1e-3

model_layer3_without_dropout = Sequential([
    Dense(10, input_dim=X_train.shape[1], activation='relu', kernel_initializer='lecun_normal', kernel_regularizer=l2(decay)),
    Dense(y_train.shape[1], kernel_initializer='he_normal', kernel_regularizer=l2(decay))
    ])

model_layer4_without_dropout = Sequential([
    
    Dense(50, input_dim=X_train.shape[1], activation='relu', kernel_initializer='lecun_normal', kernel_regularizer=l2(decay)),
    
    Dense(50, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(decay)),
    
    Dense(y_train.shape[1], kernel_initializer='he_normal', kernel_regularizer=l2(decay))
    ])

model_layer4_with_dropout = Sequential([
    
    Dense(50, input_dim=X_train.shape[1], activation='relu', kernel_initializer='lecun_normal', kernel_regularizer=l2(decay)),
    
    Dropout(rate=0.2),
    
    Dense(50, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(decay)),
    
    Dropout(rate=0.2),
    
    Dense(y_train.shape[1], kernel_initializer='he_normal', kernel_regularizer=l2(decay))
    ])
    
model_layer5_without_dropout = Sequential([
    
    Dense(50, input_dim=X_train.shape[1], activation='relu', kernel_initializer='lecun_normal', kernel_regularizer=l2(decay)),
    
    Dense(50, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(decay)),
    
    Dense(50, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(decay)),
    
    Dense(y_train.shape[1], kernel_initializer='he_normal', kernel_regularizer=l2(decay))
    ])

model_layer5_with_dropout = Sequential([
    
    Dense(50, input_dim=X_train.shape[1], activation='relu', kernel_initializer='lecun_normal', kernel_regularizer=l2(decay)),
    
    Dropout(rate=0.2),
    
    Dense(50, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(decay)),
    
    Dropout(rate=0.2),
    
    Dense(50, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(decay)),
    
    Dropout(rate=0.2),
    
    Dense(y_train.shape[1], kernel_initializer='he_normal', kernel_regularizer=l2(decay))
    ])
    
    
model_layer4_without_dropout.compile(optimizer=optimizers.SGD(lr=lr),
                  loss='mse')

model_layer4_with_dropout.compile(optimizer=optimizers.SGD(lr=lr),
                  loss='mse')

model_layer5_without_dropout.compile(optimizer=optimizers.SGD(lr=lr),
                  loss='mse')

model_layer5_with_dropout.compile(optimizer=optimizers.SGD(lr=lr),
                  loss='mse')

models = [model_layer3_without_dropout ,model_layer4_without_dropout, model_layer4_with_dropout, model_layer5_without_dropout, model_layer5_with_dropout]
models_str = ['model_layer3_without_dropout' ,'model_layer4_without_dropout', 'model_layer4_with_dropout', 'model_layer5_without_dropout', 'model_layer5_with_dropout']

val_loss = [[] for i in range(5)]

for index, m in enumerate(models):
    
    print('*'*10, models_str[index], '*'*10,)
    
    m.compile(optimizer=optimizers.SGD(lr=lr),
                  loss='mse')
    
    epoch_interval_val_loss=[100]
    
    for i in range(epochs//epochs_interval):
        
        h = m.fit(X_train, y_train, batch_size=batch_size, epochs=epochs_interval, verbose=0, validation_data=(X_test, y_test), shuffle=True)
        
    
        val_loss[index].extend(h.history['val_loss'])
        
        # for early stop
        epoch_interval_val_loss.append(h.history['val_loss'][-1])
        print(epochs_interval*(i+1),epoch_interval_val_loss[-1])

        if epoch_interval_val_loss[-1] * (1+early_stop_threshold) > epoch_interval_val_loss[-2]:
            break
            
        

plt.figure()
for i in range(5):
    plt.plot(val_loss[i] )
plt.title('mse')
plt.xlabel('epoch')
plt.legend(models_str, loc='upper left')
plt.savefig(os.path.join('others', 'plot', 'q4', 'model_comparision'))
