#!/bin/python3
import keras
keras.__version__

from keras.datasets import boston_housing
(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()
###########################normalization ?#####################################
mean=train_data.mean(axis=0)
train_data-=mean
std_div=train_data.std(axis=0)
train_data/=std_div

test_data-=mean
test_data/=std_div

##################################################################################

from keras import models
from keras import layers
##################################################################
def build_model():
	model=models.Sequential()
	model.add(layers.Dense(16,activation='relu',input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(16,activation='relu'))
	model.add(layers.Dense(1))####one interger output?
	model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
	return model
#################################################################

import numpy as np
from keras import backend as K
k = 4
K.clear_session()


all_mae_histories=[]


num_val_samples = len(train_data) // k
num_epochs = 260
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
	print('processing fold #', i)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


model2=build_model()

model2.fit(train_data,train_targets,epochs=80,batch_size=16,verbose=0)
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)















