#!/bin/python3

from keras.preprocessing.image import load_img,img_to_array
target_image_path='test-durham.jpg'
style_reference_image_path='test-monet.jpg'


width,height=load_img(target_image_path).size
img_height=400
img_width=int(width*img_height/height)
#####################################################
import numpy as np
from keras.applications import vgg19



def preprocess_image(image_path):
	img=load_img(image_path,target_size=(img_height,img_width))
	img=img_to_array(img)
	img=np.expand_dims(img,axis=0)
	img=vgg19.preprocess_input(img)
	return img


###############################################################
def deprocess_image(x):
	x[:,:,0]+=103.939
	x[:,:,1]+=116.779
	x[:,:,2]+=123.68
	x=x[:,:,::-1]
	x=np.clip(x,0,255).astype('uint8')
	return x



from keras import backend as  K

target_image=K.constant(preprocess_image(target_image_path))
style_reference_image=K.constant(preprocess_image(style_reference_image_path))
combination_image=K.placeholder((1,img_height,img_width,3))

input_tensor=K.concatenate([target_image,style_reference_image,combination_image],axis=0)

modelx=vgg19.VGG19(input_tensor=input_tensor,weights='imagenet',include_top=False)
print('____________MODEL loaded ________________')




##########################################################################
def content_loss(base,combination):
	return K.sum(K.square(combination-base))
##########################################################################

def gram_matrix(x):
	features=K.batch_flatten(K.permute_dimensions(x,(2,0,1)))
	gram=K.dot(features,K.transpose(features))
	return gram


def style_loss(style,combination):
	S=gram_matrix(style)
	C=gram_matrix(combination)
	channels=3
	size=img_height*img_width
	return K.sum(K.square(S-C))/(4.*(channels**2)*(size**2))



def total_variation_loss(x):
	a=K.square(x[:,:img_height-1,:img_width-1,:]-x[:,1:,:img_width-1,:])
	b=K.square(x[:,:img_height-1,:img_width-1,:]-x[:,:img_height-1,1:,:])
	return K.sum(K.pow(a+b,1.25))




################################################################################
output_dict=dict([(layer.name,layer.output)for layer in modelx.layers])
content_layer='block5_conv2'
style_layers=['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

total_variation_weight=1e-4
style_weight=1.
content_weight=0.025

loss=K.variable(0.)
layer_features=output_dict[content_layer]
target_image_features=layer_features[0,:,:,:]
combination_features=layer_features[2,:,:,:]
loss+=content_weight*content_loss(target_image_features,combination_features)




for layer_name in style_layers:
	layer_features=output_dict[layer_name]
	style_reference_features=layer_features[1,:,:,:]
	combination_features=layer_features[2,:,:,:]
	s1=style_loss(style_reference_features,combination_features)
	loss+=(style_weight/len(style_layers))*s1


import tensorflow as tf
loss+=total_variation_weight*total_variation_loss(combination_image)

tf.compat.v1.disable_eager_execution()
grads=K.gradients(1,combination_image)[0]








































































































