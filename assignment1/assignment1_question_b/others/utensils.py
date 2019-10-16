import numpy as np
import pandas as pd

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

def load_data(index=0):
    if index==0:
        return np.load('./npy/X_train.npy'), np.load('./npy/X_test.npy'), np.load('./npy/y_train.npy'), np.load('./npy/y_test.npy')

    elif index==1:
        return np.load('./npy/X_train_removed1.npy'), np.load('./npy/X_test_removed1.npy'), np.load('./npy/y_train_removed1.npy'), np.load('./npy/y_test_removed1.npy')

    elif index==2:
        return np.load('./npy/X_train_removed2.npy'), np.load('./npy/X_test_removed2.npy'), np.load('./npy/y_train_removed2.npy'), np.load('./npy/y_test_removed2.npy')



