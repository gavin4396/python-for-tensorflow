#!/bin/python3

import keras
keras.__version__

from keras.datasets import reuters

(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)

len(train_data)

import numpy as np

######################vetorize function###########
def vectorize_sequences(all_text,dimension=10000):
	results=np.zeros((len(all_text),dimension))
	for i,sequence in enumerate(all_text):
		results[i,sequence]=1.0
	return results
##################################################

x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)

####################one hot coding for y###########
def to_one_hot(labels,dimension=46):
	results=np.zeros((len(labels),dimension))
	for i,label in enumerate(labels):
		results[i,label]=1.
	return results
####################################################

y_train=to_one_hot(train_labels)
y_test=to_one_hot(test_labels)

####################################################
from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))
# model compile
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
######################################################

x_val=x_train[:1000]
rest_x_train=x_train[1000:]








y_val=y_train[:1000]
rest_y_train=y_train[1000:]

############### train the model and validation#########
process=model.fit(rest_x_train,rest_y_train,epochs=10,batch_size=512,validation_data=(x_val,y_val))


##############check the plots##########################
import matplotlib.pyplot as plt
loss=process.history['loss']
val_loss=process.history['val_loss']

epochs=range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='train loss')
plt.plot(epochs,val_loss,'b',label='validation loss')
plt.title('training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

acc=process.history['accuracy']
val_acc=process.history['val_accuracy']
plt.plot(epochs,acc,'bo',label='train accuracy')
plt.plot(epochs,val_acc,'b',label='validation acc')
plt.title('training and validation acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.show()
########################plot clear##########################

plt.clf()

#####################evaluation/testing part################
results=model.evaluate(x_test,y_test)
############################################################

####################prediction new data#####################
x_test=x_test[:500]
prediction=model.predict(x_test)
prediction.shape
prediction[1]
np.sum(prediction[1])#############check all possibility summation=1
np.argmax(prediction[1])


"""
Wrapping up

Here's what you should take away from this example:

   * If you are trying to classify data points between N classes, your network should end with a Dense layer of size N.
    *In a single-label, multi-class classification problem, your network should end with a softmax activation, so that it will output a probability distribution over the N output classes.
    *Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the network, and the true distribution of the targets.
    *There are two ways to handle labels in multi-class classification: Encoding the labels via "categorical encoding" (also known as "one-hot encoding") and using categorical_crossentropy as your loss function. Encoding the labels as integers and using the sparse_categorical_crossentropy loss function.
    If you need to classify data into a large number of categories, then you should avoid creating information bottlenecks in your network by having intermediate layers that are too small.
"""
####################################this is integers and using the sparse_categorical_crossentropy loss function######################
model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))
# model compile
model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
######################################################

x_val=x_train[:1000]
rest_x_train=x_train[1000:]








y_val=train_labels[:1000]
rest_y_train=train_labels[1000:]

############### train the model and validation#########
process=model.fit(rest_x_train,rest_y_train,epochs=10,batch_size=512,validation_data=(x_val,y_val))



