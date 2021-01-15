#!/bin/python3

import keras
keras.__version__

###################preparing the dataset#########################################################
import os,shutil

#path to the directory where the dataset was uncompressed
original_dataset_dir='/home/yang/dog_cat_dataset/dogs-vs-cats/train'

#path where will store the small datasets
base_dir='/home/yang/dog_cat_small'
os.mkdir(base_dir)
#######################################################
#Direction for training
train_dir=os.path.join(base_dir,'train')
os.mkdir(train_dir)
#direction for validation
validation_dir=os.path.join(base_dir,'validation')
os.mkdir(validation_dir)
#direction for testing
test_dir=os.path.join(base_dir,'test')
os.mkdir(test_dir)

#######################################################
#Cat training Direction
train_cats_dir=os.path.join(train_dir,'cats')
os.mkdir(train_cats_dir)
#Dog training Direction
train_dogs_dir=os.path.join(train_dir,'dogs')
os.mkdir(train_dogs_dir)

#Cat validation Direction
validation_cats_dir=os.path.join(validation_dir,'cats')
os.mkdir(validation_cats_dir)
#Dog validation Direction
validation_dogs_dir=os.path.join(validation_dir,'dogs')
os.mkdir(validation_dogs_dir)

#Cat testing Direction
test_cats_dir=os.path.join(test_dir,'cats')
os.mkdir(test_cats_dir)
#Dog testing Direction
test_dogs_dir=os.path.join(test_dir,'dogs')
os.mkdir(test_dogs_dir)


#111	 copy 1000 cat images to train_cats_dir
fnames=['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src=os.path.join('home/yang/dog_cat_dataset/dogs-vs-cats/train_25000',fname)
	dst=os.path.join('home/yang/dog_cat_small/train/cat',fname)
	shutil.copyfile(src,dst)

#222 	copy 1000 dog images to train_dogs_dir
fnames=['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src=os.path.join(original_dataset_dir,fname)
	dst=os.path.join(train_dogs_dir,fname)
	shutil.copyfile(src,dst)

#333 	copy 500 cat images to validation_cats_dir
fnames=['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
	src=os.path.join(original_dataset_dir,fname)
	dst=os.path.join(validation_cats_dir,fname)
	shutil.copyfile(src,dst)

#444 	copy 500 dog images to validation_dogs_dir
fnames=['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
	src=os.path.join(original_dataset_dir,fname)
	dst=os.path.join(validation_dogs_dir,fname)
	shutil.copyfile(src,dst)

#555 	copy 500 cat images to test_cats_dir
fnames=['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
	src=os.path.join(original_dataset_dir,fname)
	dst=os.path.join(test_cats_dir,fname)
	shutil.copyfile(src,dst)

#666 	copy 500 dog images to test_dogs_dir
fnames=['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
	src=os.path.join(original_dataset_dir,fname)
	dst=os.path.join(test_dogs_dir,fname)
	shutil.copyfile(src,dst)
###################################################################
print('total trainning cat images:',len(os.listdir(train_cats_dir)))
print('total trainning dog images:',len(os.listdir(train_dogs_dir)))
print('total validation_cats_dir:',len(os.listdir(validation_cats_dir)))
print('total validation_dogs_dir:',len(os.listdir(validation_dogs_dir)))
print('total test_cats_dir:',len(os.listdir(test_cats_dir)))
print('test_dogs_dir',len(os.listdir(test_dogs_dir)))
###################################################################

###################convnet and layers setup########################
from keras import layers
from keras import models
model1=models.Sequential()



model1.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Conv2D(64,(3,3),activation=('relu')))
model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Conv2D(128,(3,3),activation=('relu')))
model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Conv2D(128,(3,3),activation=('relu')))
model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Flatten())
model1.add(layers.Dense(512,activation='relu'))
model1.add(layers.Dense(1,activation='sigmoid'))

from keras import optimizers

model1.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

########################################################################################################
#####################################Data preparing####################################################

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
validation_generator=test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='binary')


for data_batch,labels_batch in train_generator:
	print('data batch shape',data_batch.shape)
	print('labels batch shape',labels_batch)
	break

history=model1.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50)




























