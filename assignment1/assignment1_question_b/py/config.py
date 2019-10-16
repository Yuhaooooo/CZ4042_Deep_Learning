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

csv_file = '/Users/heyuhao/Documents/GitHub/school_project/cz4042/assignment1_question_b/admission_predict.csv'

df = pd.read_csv(csv_file, index_col=[0])

data = df.values

X = data[:, :-1]
y = data[:, -1].reshape(-1,1)

def scale(X, decimals):
    return np.round((X - np.mean(X, axis=0))/ np.std(X, axis=0), decimals=decimals)

X = scale(X, 4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

