#!/bin/python3
from keras.applications import VGG16
conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
# weight is the weight passed into model for convnet, Include_top is if you want to include the dense classifier
conv_base.summary()

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_dir='/home/yang/dog_cat_small/train'
val_dir='/home/yang/dog_cat_small/validation'
test_dir='/home/yang/dog_cat_small/test'

datagen=ImageDataGenerator(rescale=1./255)
batch_size=20

#######################################################feature extraction function########################|
def extract_feature(directory,samples_count):
	feature_tensor=np.zeros(shape=(samples_count,4,4,512))
	label_tensor=np.zeros(shape=samples_count)
	generator=datagen.flow_from_directory( directory,target_size=(150,150),batch_size=batch_size,class_mode='binary')
	i=0;
	for inputs_batch,label_batch in generator:
		feature_batch=conv_base.predict(inputs_batch)##predict here in every batch lop
		feature_tensor[i*batch_size:(i+1)*batch_size]=feature_batch
		label_tensor[i*batch_size:(i+1)*batch_size]=label_batch
		i+=1
		print(i)
		if i*batch_size>=samples_count:
			break
	return feature_tensor,label_tensor	



train_features,train_labels=extract_feature(train_dir,2000)
validation_features,validation_labels=extract_feature(val_dir,1000)
test_features,test_labels=extract_feature(test_dir,1000)
train_features=np.reshape(train_features,(2000,4*4*512))
validation_features=np.reshape(validation_fetures,(1000,4*4*512))
test_features=np.reshape(test_features,(1000,4*4*512))




from keras import models
from keras import layers
from keras import optimizers

model=models.Sequential()
model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
model.add)layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

history=model.fit(train_features,train_labels,epochs=30,batch_size=16,validation_data=(validation_features,validation_labels))





