#!bin/python3
from keras.models import load_model
import numpy as np
model=load_model('cats_and_dogs_small.h5')

img_path='/home/yang/dog_cat_small/test/cats/cat.1502.jpg'
from keras.preprocessing import image
img=image.load_img(img_path,target_size=(150,150))
img_tensor=image.img_to_array(img)
img_tensor=np.expand_dims(img_tensor,axis=0)
img_tensor/=255
#########################################################
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()

from keras import models

layer_output=[layer.output for layer in model.layers[:8]]
activation_model=models.Model(inputs=model.input,outputs=layer_output)
activations=activation_model.predict(img_tensor)

layers_activation_1st=activations[0]
print(layers_activation_1st.shape)
tensor_1layer_4chan=layers_activation_1st[0,:,:,4]
plt.imshow(tensor_1layer_4chan)
plt.show()

layer_names=[]
for layer in model.layers[:8]:
	layer_names.append(layer.name)

image_per_row=16
for layer_name,layer_activation in zip(layer_names,activations):
	n_features=layer_activation.shape[-1]
	size=layer_activation.shape[1]
	n_cols=n_features//image_per_row
	
	display_grid=np.zeros((size*n_cols,image_per_row*size))
	
	for col in range(n_cols):
		for row in range (image_per_row):
			channel_image=layer_activation[0,:,:,col*image_per_row+row]
#########################################################################3


			channel_image*=64
			channel_image+=128
			channel_image=np.clip(channel_image,0,255).astype('uint8')
			display_grid[col*size:(col+1)*size,row*size:(row+1)*size]=channel_image
		
	scale=1. / size
	plt.figure(figsize=(scale*display_grid.shape[1],scale*display_grid.shape[0]))
	plt.title(layer_name)
	plt.grid(False)
	plt.imshow(display_grid,aspect='auto',cmap='viridis')
	plt.savefig(layer_name,dpi=400,bbox_inches='tight')
	plt.show()
















