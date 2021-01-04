#!/bin/python3

import keras 
keras.__version__

from keras.datasets import imdb
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000);
import numpy as np

max([max(sequence) for sequence in train_data])


def vectorize_sequence(all_text,dimension=10000):
	one_hot_matrix=np.zeros((len(all_text),dimension))
	for i,word_index in enumerate(all_text):
		one_hot_matrix[i,word_index]=1.0
	return one_hot_matrix


train_data2=train_data[0:10000,]
train_labels2=train_labels[0:10000]
test_data2=test_data[0:10000,]
test_labels2=test_labels[0:10000]

x_train=vectorize_sequence(train_data2)
y_train=np.asarray(train_labels2).astype('float32')
x_test=vectorize_sequence(test_data2)
y_test=np.asarray(test_labels2).astype('float32')


print('vetorize job done')

from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,batch_size=512)

results=model.evaluate(x_test,y_test)
