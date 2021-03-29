#!/bin/python3


####################redistribution the text, so that we could get interesting paper#################

import numpy as np

def reweight_distribution(original_distribution,temperature):
#original_distribution look like 0.1 0.2 0.3 0.2 0.3 1D tensor add up with 1
	distribution=np.log(original_distribution)/temperature
	distribution=np.exp(distribution)
	distribution=distribution/np.sum(distribution)
	return distribution

#######################################################################################################
import keras 
path=keras.utils.get_file('nietzsche.txt',origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text=open(path).read().lower()
print('Corpus length:',len(text))

########################################################################################################
maxlen=20
step=8
sentences=[]
next_chars=[]

for i in range(0,len(text)-maxlen,step):
	sentences.append(text[i:i+maxlen])
	next_chars.append(text[i+maxlen])
	
print('number of sequences:',len(sentences))

chars=sorted(list(set(text)))
print(len(chars),'unique characters')
char_indices=dict((char,chars.index(char)) for char in chars)

print('Vectorization...')
x=np.zeros((len(sentences),maxlen,len(chars)),dtype=np.bool)
y=np.zeros((len(sentences),len(chars)),dtype=np.bool)

for i ,sentence in enumerate(sentences):
	for t,char in enumerate(sentence):
		x[i,t,char_indices[char]]=1
	y[i,char_indices[next_chars[i]]]=1


#########################################################################################################
#constructor an LSTM network processing sequences########################################################
from keras import layers

modelx=keras.models.Sequential()
modelx.add(layers.LSTM(32,input_shape=(maxlen,len(chars))))
modelx.add(layers.Dense(len(chars),activation='softmax'))

optimizer_1=keras.optimizers.RMSprop(lr=0.01)
modelx.compile(loss='categorical_crossentropy',optimizer=optimizer_1)

###########################################################################################################
def sample(preds,temperature):
	preds=np.asarray(preds).astype('float64')
	preds=np.log(preds)/temperature
	preds_exp=np.exp(preds)
	preds=preds_exp/np.sum(preds_exp)
	probability=np.random.multinomial(1,preds,1)
	Final=np.argmax(probability)
	return Final
###########################################################################################################

import random 
import sys

modelx.fit(x,y,batch_size=32,epochs=1)
	




#####################################################################################################################
start_index=random.randint(0,len(text)-maxlen-1)
generated_index=text[start_index:start_index+maxlen]
print('SEED:'+generated_index)

temperature=0.1












































