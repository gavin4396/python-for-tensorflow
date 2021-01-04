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
y_tt=np.asarray(test_labels2).astype('float32')


print('vetorize job done')

from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

x_val=x_train[:500]
partial_x_train=x_train[500:]#remain x for trainning
y_val=y_train[:500]
partial_y_train=y_train[500:]#remain y for trainning

process=model.fit(partial_x_train,partial_y_train,epochs=5,batch_size=512,validation_data=(x_val,y_val))

history_dict=process.history
history_dict.keys()

import matplotlib.pyplot as plt

history_dict=process.history
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']

epochs=range(1,len(loss_values)+1)

plt.plot(epochs,loss_values,'bo',label='train losses')
plt.plot(epochs,val_loss_values,'b',label='validation losses')
plt.title('trainning loss VS validation loss')
plt.xlabel('epoch times')
plt.ylabel('loss value')
plt.legend()
plt.show()

plt.clf()
acc=history_dict['accuracy']
val_acc=history_dict['val_accuracy']

plt.plot(epochs,acc,'bo')
plt.plot(epochs,val_acc,'b')
plt.legend()
plt.show()



