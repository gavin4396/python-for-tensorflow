#!/bin/python3
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential
max_features=10000
max_len=600

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
x_train=[x[::-1]for x in x_train]
x_test=[x[::-1]for x in x_test]

x_train=sequence.pad_sequences(x_train,maxlen=max_len)	#|pading the sequence of each comment
x_test=sequence.pad_sequences(x_test,maxlen=max_len)		#|

model=Sequential()
model.add(layers.Embedding(max_features,128))#a 128 dimensional embedding layer
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=0.1)

