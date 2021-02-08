#!/bin/python3
from keras.models import Sequential
from keras.layers import Embedding,SimpleRNN


from keras.datasets import imdb
from keras.preprocessing import sequence

MAX_FEATURES =10000
maxlen=500
batch_size=32

print('data loading')

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=MAX_FEATURES)

print(len(x_train),' trainning samples')
print(len(x_test),'testing samples')

x_pad_train=sequence.pad_sequences(x_train,maxlen=maxlen)#padding all the train sequence to a fixed length 1D array.
x_pad_test=sequence.pad_sequences(x_test,maxlen=maxlen)
#################################################################
######pad are len(samples) x 500 []##############################

from keras.layers import Dense

model=Sequential()
model.add(Embedding(MAX_FEATURES,32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(x_pad_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

###########################################################################
#######################LSTM on RNN##########################################
from keras.layers import LSTM

model=Sequential()
model.add(Embedding(MAX_FEATURES,32))
model.add(LSTM(16))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(x_pad_train,y_train,epochs=5,batch_size=128,validation_split=0.2)


































