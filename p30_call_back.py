#!/bin/python3
import keras

callbacks_list=[keras.callbacks.EarlyStopping(monitor='acc',patience=1),\
				keras.callbacks.ModelCheckpoint(filepath='my_model.h5',monitor='val_loss',save_best_only=Ture,)]

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc']

model.fit(x,y,epochs=10,batch_size=32,callbacks=callbacks_list,validation_data=(x_val,y_val))
