#!/bin/python3

from keras.models import Sequential,Model
from keras import layers
from keras import Input

seq_model=Sequential()
seq_model.add(layers.Dense(32,activation='relu',input_shape=(64,)))
seq_model.add(layers.Dense(16,activation='relu'))
seq_model.add(layers.Dense(10,activation='softmax'))
seq_model.summary()
#functional_API

Input_tensor=Input(shape=(64,))
x=layers.Dense(16,activation='relu')(Input_tensor)
x=layers.Dense(32,activation='relu')(x)
x=layers.Dense(10,activation='softmax')(x)

model=Model(Input_tensor,x)
model.summary()


model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
import numpy as np
x_train=np.random.random((1000,64))
y_train=np.random.random((1000,10))

model.evaluate(x_train,y_train)





################################################################################


