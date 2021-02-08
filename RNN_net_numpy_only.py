#!/bin/python3
import numpy as np
time_step=100
input_feature=32

output_feature=64

inputs=np.random.random((time_step,input_feature))
state_t=np.zeros((output_features,))

W=np.random.random((output_feature,input_feature))
U=np.random.random((output_feature,output_feature))
b=np.random.random((output_feature))

successive_output=[]

for input_i in inputs
	output_i=np.dot(W,input_i)+np.dot(U,state_t)+b#W x input_i + U x state_i +b
	successive_output.append(output_i)
	state_i=output_i
final_output_sequence=np.stack(successive_output,axis=0)
