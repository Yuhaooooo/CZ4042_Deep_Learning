import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import re
import string
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))
import datetime
import matplotlib.pyplot as plt
import os
import sys

no_words = 34016
no_epochs = 3000
batch_size = 128
lr = 0.01

MAX_DOCUMENT_LENGTH = 100
EMBEDDING_SIZE = 20
MAX_LABEL = 14


class TimeRecoder():
    start_time = 0
    end_time = 0
    
    def start(self):
        self.start_time = datetime.datetime.now()
        print('\n\n', '-'*10, ' START ', '-'*10, self.start_time, '\n\n')
        
    def end(self):
        self.end_time = datetime.datetime.now()
        print('\n\n', '-'*10, ' END ', '-'*10, self.end_time, '\n\n')
        self.checkTimeSpan()
        
    def checkTimeSpan(self):
        print('\n\n', '-'*10, ' TAKES ', '-'*10, self.end_time - self.start_time, '\n\n')


def clean_data(csv_path):

	global no_words

	dir_path = os.path.dirname(os.path.realpath(__file__))

	MAX_DOCUMENT_LENGTH=100

	df_train = pd.read_csv(os.path.join(csv_path, 'train_medium.csv'), names=['class', '_', 'text'], usecols=['class', 'text'])
	df_test = pd.read_csv(os.path.join(csv_path, 'test_medium.csv'), names=['class', '_', 'text'], usecols=['class', 'text'])

	train_data = df_train.values
	test_data = df_test.values

	def clean_text(t):
	    #lowercase
	    t= t.lower()
	    #remove number and punctuation
	    t = re.sub(r'([^\w\s])|\d','',t)
	    #remove white space
	    t = re.sub(r'\s+',' ',t)
	    #remove stopwords
	    t = ' '.join([t for t in t.split(' ') if t not in stopwords])
	    t = t.strip()
	    return t

	for pair in train_data:
	    pair[1] = clean_text(pair[1])
	for pair in test_data:
	    pair[1] = clean_text(pair[1])
	    
	train_data = np.array(train_data)
	test_data = np.array(test_data)
	    
	x_train = pd.Series(train_data.T[1])
	y_train = pd.Series(train_data.T[0])
	x_test = pd.Series(test_data.T[1])
	y_test = pd.Series(test_data.T[0])

	# char features
	char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
	x_train_char = np.array(list(char_processor.fit_transform(x_train)))
	x_test_char = np.array(list(char_processor.transform(x_test)))
	y_train_char = np.array(y_train.values)
	y_test_char = np.array(y_test.values)

	np.save(os.path.join(dir_path, '..', 'npy', 'x_train_char.npy'), x_train_char)
	np.save(os.path.join(dir_path, '..', 'npy', 'x_test_char.npy'), x_test_char)
	np.save(os.path.join(dir_path, '..', 'npy', 'y_train_char.npy'), y_train_char)
	np.save(os.path.join(dir_path, '..', 'npy', 'y_test_char.npy'), y_test_char)


	# word features
	vocab_processor=tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
	x_train_word = np.array(list(vocab_processor.fit_transform(x_train)))
	x_test_word = np.array(list(vocab_processor.transform(x_test)))
	y_train_word = np.array(y_train.values)
	y_test_word = np.array(y_test.values)
	no_words = len(vocab_processor.vocabulary_)

	np.save(os.path.join(dir_path, '..', 'npy', 'x_train_word.npy'), x_train_word)
	np.save(os.path.join(dir_path, '..', 'npy', 'x_test_word.npy'), x_test_word)
	np.save(os.path.join(dir_path, '..', 'npy', 'y_train_word.npy'), y_train_word)
	np.save(os.path.join(dir_path, '..', 'npy', 'y_test_word.npy'), y_test_word)


def main():
	clean_data(sys.argv[1])
	print('\n\n Finish processing data...\n\n')

if __name__ == '__main__':
	main()



