#!/bin/python3

import string
import numpy as np

samples=['Here is your last hope.','But I am not sure if it will work.']
characters=string.printable
token_index=dict(zip(range(1,len(characters)+1),characters))
max_length=50
results=np.zeros((len(samples),max_length,max(token_index.keys())+1))

for i,sample in enumerate(samples):
	for j,character in enumerate(sample):
		index=token_index.get(character)
		results[i,j,index]=1.


