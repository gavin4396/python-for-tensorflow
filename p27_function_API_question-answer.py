#!/bin/python3

from keras.models import Model
from keras import layers
from keras import Input


#################################
text_vocabulary_size=1000
question_vocabulary_size=1000
answer_vocabulary_size=500

#################################


text_input=Input(shape=(None,),dtype='int32',name='text')
embedded_text=layers.Embedding(text_vocabulary_size,64)(text_input)
encoded_text=layers.LSTM(32)(embedded_text)
#----------------------------------------------------------------------
question_input=Input(shape=(None,),dtype='int32',name='question')
embedded_question=layers.Embedding(text_vocabulary_size,64)(question_input)
encoded_question=layers.LSTM(32)(embedded_question)
#----------------------------------------------------------------------
concatenated=layers.concatenate([encoded_text,encoded_question],axis=-1)

answer=layers.Dense(answer_vocabulary_size,activation='softmax')(concatenated)


model=Model([text_input,question_input],answer)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics='accuracy')


#####################################################################################
import numpy as np
import keras
num_samples=256
max_length=544
 
text_train=np.random.randint(1,text_vocabulary_size,size=(num_samples,max_length))
question_train=np.random.randint(1,question_vocabulary_size,size=(num_samples,max_length))
answer_train=np.random.randint(answer_vocabulary_size,size=(num_samples))
answer_train=keras.utils.to_categorical(answer_train,answer_vocabulary_size)

##################################################################################

model.fit([text_train,question_train],answer_train,epochs=10,batch_size=32)

