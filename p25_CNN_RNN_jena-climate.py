import os 

data_dir='/home/yang/jena_climate'
fname=os.path.join(data_dir,'jena_climate_2009_2016.csv')

f=open(fname)
data=f.read()
f.close()

lines=data.split('\n')
headers=lines[0].split(',')
lines=lines[1:]

print(headers)
print(len(lines))


############################################################################################
import numpy as np

float_data=np.zeros((len(lines),len(headers)-1))
for i, line in enumerate(lines):
	values=[float(x) for x in line.split(',')[1:]]
	float_data[i,:]=values

######################################normalization data#######################################
mean=float_data[:200000].mean(axis=0)
float_data-=mean
std=float_data[:200000].std(axis=0)
float_data/=std









#############################################################################################
###############define a generator to give the requied data in case waste of memory###########
def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=128,step=6):
	if max_index is None:
		max_index=len(data)-delay-1
	i=min_index+lookback
	while 1:
		if shuffle:
			rows=np.random.randint(min_index+lookback,max_index,size=batch_size)#rows=[200128,200000,...,200512]
		else:
			if i+batch_size>=max_index:
				i=min_index+lookback
			rows=np.arange(i,min(i+batch_size,max_index))
			i+=len(rows)
		samples=np.zeros((len(rows),lookback//step,data.shape[-1]))
		targets=np.zeros((len(rows),))
		for j,row in enumerate(rows):
			indices=range(rows[j]-lookback,rows[j],step)
			samples[j]=data[indices]
			targets[j]=data[rows[j]+delay][1]
		yield samples,targets

##############################################################################################
lookback=1440
step=6
delay=144
batch_size=128

train_gen=generator(data=float_data,lookback=lookback,delay=delay,min_index=0,max_index=200000,shuffle=True,step=3,batch_size=batch_size)


val_gen=generator(data=float_data,lookback=lookback,delay=delay,min_index=200001,max_index=300000,step=3,batch_size=batch_size)


test_gen=generator(data=float_data,lookback=lookback,delay=delay,min_index=300001,max_index=None,step=3,batch_size=batch_size)

val_steps=(300000-200001-lookback)//128
##########################################################################################################################################

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

#########################################################################################################
#
#
#########################################################################################################
model=Sequential()
model.add(layers.Conv1D(32,5,activation='relu',input_shape=(None,float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32,5,activation='relu'))
model.add(layers.GRU(32,dropout=0.1,recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.summary()


model.compile(optimizer=RMSprop(),loss='mae')
model.fit_generator(train_gen,steps_per_epoch=500,epochs=5,validation_data=val_gen,validation_steps=val_steps)


















































