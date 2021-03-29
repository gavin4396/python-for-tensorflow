#! bin/python3


from keras import layers
from keras import Input

x=Input(shape=(None,),dtype='int32',name='text')

brunch_a= layers.Conv2D(128,1,activation='relu',strides=2)(x)

brunch_b=layers.Conv2D(128,1,activation='re;u')(x)
brunch_b=layers.Conv2D(128,3,activation='relu',strides=2)(brunch_b)

brunch_c=layers.Conv2D(128,3,activation='relu',strides=2)(x)
brunch_c=layers.Conv2D(128,3,activation='relu')(brunch_c)

brunch_d=layers.Conv2D(128,1,activation='relu')(x)
brunch_d=layers.Conv2D(128,3,activation='relu')(brunch_d)
brunch_d=layers.Conv2D(128,3,activation='relu',strides=2)(brunch_d)

output=layers.concatenate([brunch_a,brunch_b,brunch_c,brunch_d],axis=-1)











































