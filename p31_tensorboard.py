#!/bin/python3
import keras 
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features=10000
max_len=500

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
x_train=sequence.pad_sequences(x_train,maxlen=max_len)
x_test=sequence.pad_sequences(x_test,maxlen=max_len)
##########################modelx is a sequencial model############################
modelx=keras.models.Sequential()
modelx.add(layers.Embedding(max_features,128,input_length=max_len,name='embed'))
modelx.add(layers.Conv1D(32,7,activation='relu'))
modelx.add(layers.MaxPooling1D(5))
modelx.add(layers.Conv1D(32,7,activation='relu'))
modelx.add(layers.GlobalMaxPooling1D())
modelx.add(layers.Dense(1))
##################################################################################

modelx.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
####################to use tensorboard we would need a callback function##########
callbacks=[keras.callbacks.TensorBoard(log_dir='my_log_dir',histogram_freq=1,embeddings_freq=1)]
history=modelx.fit(x_train,y_train,epochs=20,batch_size=128,validation_split=0.1,callbacks=callbacks)
