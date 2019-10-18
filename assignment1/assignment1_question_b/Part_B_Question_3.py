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


csv_file = os.path.join('others', 'admission_predict.csv')
df = pd.read_csv(csv_file, index_col=[0])
X_train, X_test, y_train, y_test = np.load(os.path.join('others', 'npy', 'X_train.npy')), np.load(os.path.join('others', 'npy', 'X_test.npy')), np.load(os.path.join('others', 'npy', 'y_train.npy')), np.load(os.path.join('others', 'npy', 'y_test.npy'))
lr = 1e-3
decay = 1e-3
batch_size = 8
epochs=2900

def remove_first_feature():

    features = list(df.columns)[:-1]

    print('All features: {}..\n'.format(features))

    mse = dict(zip(features, [0 for i in features]))

    for i in range(len(features)):
        
        excluded_feature = features[i]
        
        print('\nRemoved {}...'.format(excluded_feature))
        print('feature kept: {}'.format([f for f in features if f!=excluded_feature]))
        
        X_train_removed, X_test_removed = np.delete(X_train, obj=i, axis=1), np.delete(X_test, obj=i, axis=1)
        
        model = Sequential([
        Dense(10, input_dim=X_train_removed.shape[1], activation='relu', kernel_initializer='lecun_normal', kernel_regularizer=l2(decay)),
        Dense(y_train.shape[1], kernel_initializer='he_normal', kernel_regularizer=l2(decay))
        ])

        model.compile(optimizer=optimizers.SGD(lr=lr),
                    loss='mse')

        h = model.fit(X_train_removed, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_test_removed, y_test), shuffle=True)
        
        mse[excluded_feature] = h.history['val_loss'][-1] 
    
        print('mse: {}\n\n'.format(h.history['val_loss'][-1]))

    feature_to_be_removed = min(mse, key=mse.get)
    print('\n\n\n\nThe first feature removed is {}\n\n\n\n'.format(feature_to_be_removed))

    return features.index(feature_to_be_removed)

def remove_second_feature(feature1_index):

    features = list(df.columns)[:-1]
    features.pop(feature1_index)

    mse = dict(zip(features, [0 for i in features]))
    
    for i in range(len(features)):
        
        excluded_feature = features[i]
        
        print('\nRemoved {}...'.format(excluded_feature))
        print('feature kept: {}'.format([f for f in features if f!=excluded_feature]))
        
        X_train_removed, X_test_removed = np.delete(X_train, obj=i, axis=1), np.delete(X_test, obj=i, axis=1)
        
        model = Sequential([
        Dense(10, input_dim=X_train_removed.shape[1], activation='relu', kernel_initializer='lecun_normal', kernel_regularizer=l2(decay)),
        Dense(y_train.shape[1], kernel_initializer='he_normal', kernel_regularizer=l2(decay))
        ])

        model.compile(optimizer=optimizers.SGD(lr=lr),
                    loss='mse')

        h = model.fit(X_train_removed, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_test_removed, y_test), shuffle=True)
        
        mse[excluded_feature] = h.history['val_loss'][-1]
        
        print('mse: {}\n\n'.format(h.history['val_loss'][-1]))

    feature_to_be_removed = min(mse, key=mse.get)
    print('\n\n\n\nThe second feature removed is {}\n\n\n\n'.format(feature_to_be_removed))

    return features.index(feature_to_be_removed)



print('\n\n', '-'*10, '  Question 3(a)  ', '-'*10, '\n\n')
feature1_removed_index = remove_first_feature()
X_train, X_test = np.delete(X_train, obj=feature1_removed_index, axis=1), np.delete(X_test, obj=feature1_removed_index, axis=1)
print('\n\n', '-'*10, '  Question 3(b)  ', '-'*10, '\n\n')
feature2_removed_index = remove_second_feature(feature1_removed_index)
X_train, X_test = np.delete(X_train, obj=feature2_removed_index, axis=1), np.delete(X_test, obj=feature2_removed_index, axis=1)

np.save(os.path.join('others', 'npy', 'X_train2.npy'), X_train)
np.save(os.path.join('others', 'npy', 'X_test2.npy'), X_test)

print(X_train.shape, X_test.shape)
print('The new train and test dataset is save in others/npy/X_train2.npy and others/npy/X_test2.npy')