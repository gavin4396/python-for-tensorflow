#!/bin/python3
import numpy as np

samples=['The cat sit on the table.','The dog walk on the yard.']

#############create a diction first###############################
token_index={}
for sample in samples:
	for word in sample.split():
		if word not in token_index:
			token_index[word]=len(token_index)+1
######################################################
max_length=8
results=np.zeros(shape=(len(samples),max_length,max(token_index.values())+1))

i=0
en_samples=enumerate(samples)
for i,sample in en_samples:
	for j,word in list(enumerate(sample.split())) [:max_length]:#list is to subscript from 0 to MAX_LENGTH
		index=token_index.get(word)####which number this word correspond to
		results[i,j,index]=1.











