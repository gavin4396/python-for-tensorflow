#!/bin/python3

from keras.datasets import imdb
from keras.preprocessing import sequence

max_features=10000
max_len=500
print('Data loading....................................................')
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
print(len(x_train),' train sequences')
print(len(y_train),'test sequences')
x_train_pad=sequence.pad_sequences(x_train,maxlen=max_len)
x_test_pad=sequence.pad_sequences(x_test,maxlen=max_len)


from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model=Sequential()
model.add(layers.Embedding(max_features,128,input_length=max_len))#128 dimentional
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer=RMSprop(lr=1e-4),loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(x_train_pad,y_train,epochs=5,batch_size=256,validation_split=0.2)









