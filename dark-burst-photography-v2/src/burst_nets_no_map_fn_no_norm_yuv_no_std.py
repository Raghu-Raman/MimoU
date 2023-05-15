# set_nets.py: Ahmet Serdar Karadeniz
# description: Networks for set based model.

from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import numpy as np
import glob
import numpy as np
#import tensorflow_addons as tfa
import math
import dbputils

class MyMapConv2dLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyMapConv2dLayer, self).__init__()

    def call(self, inputs, dims, ksize, activation_fn):
        print("inputs.shape of map layer is, ", inputs.shape)
        covLayer=tf.keras.layers.Conv2D(filters=dims,kernel_size=ksize, activation=activation_fn, padding="same")(inputs)
        return covLayer
class MyMapMaxpoolingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyMapMaxpoolingLayer, self).__init__()

    def call(self, results):
        return tf.nn.max_pool2d(results, [2, 2], strides=2, padding='SAME')



def lrelu(x):
	return tf.maximum(x * 0.2, x)

def upsample(input, s=2, nn=False):
	sh = tf.shape(input=input)
	newShape = s * sh[1:3]
	if nn == False:
		output = tf.image.resize(input, newShape, method=tf.image.ResizeMethod.BILINEAR)
	else:
		output = tf.image.resize(input, newShape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	return output

def upsample_and_concat(x1, x2, output_channels, in_channels, is_fine=True, block_idx=0):
	pool_size = 2
	# print("x1 shape is: ",x1.shape)
	# print("x2 shape is: ",x2.shape)
	if is_fine == True:
		name = "deconv_fine_0"
		if block_idx > 0:
			name = name+"_%d"%block_idx

		deconv = tf.keras.layers.Conv2DTranspose(filters= output_channels, kernel_size=2, strides=[pool_size, pool_size], padding="SAME")(x1)
		deconv_output =  tf.keras.layers.concatenate([deconv, x2], axis=3)

	return deconv_output

def encode_block_light(inputs, dims, activation_fn, block_idx, max_pool=True, normalizer_fn=None,  module_name="original", ksize1=3, ksize2=3, use_center=True):
	if module_name != "original":
		module_name = "_"+module_name
	else:
		module_name = ""
	encs = tf.keras.layers.Conv2D(filters=dims, kernel_size=ksize1, activation='relu', padding="same")(inputs)
	global_pool = encs

	results = encs
	if max_pool == True:
		max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
												   strides=(2, 2), padding='SAME')
		results = max_pool_2d(results)
	
	return results, encs, global_pool

def decode_block(inputs, inputs_early, out_channels, in_channels, activation_fn, block_idx, module_name, ksize=3):
	if module_name != "original":
		module_name = "_"+module_name
	else:
		module_name = ""

	up = upsample_and_concat(inputs, inputs_early, out_channels, in_channels, block_idx=block_idx-6)
	conv = tf.keras.layers.Conv2D(filter=out_channels, kernel_size=ksize, activation=activation_fn)(up)
	conv = tf.keras.layers.Conv2D(filter=out_channels, kernel_size=ksize, activation=activation_fn)(conv)
	conv.set_shape([None, None, None, None, out_channels])
	global_pool = tf.reduce_max(input_tensor=conv, axis=0)

	return conv, global_pool

def fine_mimo(inp_y1, inp_uv1, inp_y2,inp_uv2,inp_y3, inp_uv3, out_channels=6, dims=8, nres_block=2, normalizer_fn=None, module_name="fine", demosaic=True, use_center=True, use_noise_map=False):
	print("inp_y0.shape",inp_y1.shape)
	print("inp_uv0.shape",inp_uv1.shape)
	inp_y1Temp = tf.keras.layers.Conv2D(filters=4, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(inp_y1)
	inp_uv1Temp = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(inp_uv1)
	inputs_ = tf.keras.layers.concatenate([inp_y1Temp, inp_uv1Temp], axis=3)
	print("Input to the first layer", inputs_.shape)
	
	pool1s, conv1s, conv1 = encode_block_light(inputs_, dims,  activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=1, use_center=use_center)
	inp_y2 = tf.keras.layers.Conv2D(filters=4, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(inp_y2)
	inp_uv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(inp_uv2)
	inputs2_ = tf.keras.layers.concatenate([inp_y2, inp_uv2], axis=3)
	combine1and2 = tf.keras.layers.concatenate([pool1s, inputs2_], axis=3)
	pool2s, conv2s, conv2 = encode_block_light(combine1and2, dims*2,  activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=1, use_center=use_center)
	
	inp_y3 = tf.keras.layers.Conv2D(filters=4, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(inp_y3)
	inp_uv3 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(inp_uv3)
	inputs3_ = tf.keras.layers.concatenate([inp_y3, inp_uv3], axis=3)
	combine2and3 = tf.keras.layers.concatenate([pool2s, inputs3_], axis=3)
	pool3s, conv3s, conv3 = encode_block_light(combine2and3, dims*4,  activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=1, use_center=use_center)

	pool4s, conv4s, conv4 = encode_block_light(pool3s, dims*8,  activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=1, use_center=use_center)
	pool5s, conv5s, conv5 = encode_block_light(pool4s, dims*16,  activation_fn=lrelu, normalizer_fn=normalizer_fn, module_name=module_name, block_idx=1, use_center=use_center)

	net = conv5
	for i in range(nres_block):
		temp = net
		net = tf.keras.layers.Conv2D(filters=dims*16, kernel_size=3, activation='relu', padding="same")(net)
		net = tf.keras.layers.Conv2D(filters=dims*16, kernel_size=3, activation=None, padding="same")(net)
		#net = se_block(net, dims*16, block_idx=i)
		net = net + temp

	net = tf.keras.layers.Conv2D(filters=dims*16, kernel_size=3, activation=None, padding="same")(net)
	conv5 = net + conv5

	up6 = upsample_and_concat(conv5, conv4, dims*8, dims*16, is_fine=True, block_idx=0)
	conv6 = tf.keras.layers.Conv2D(filters=dims*8, kernel_size=3, activation='relu', padding="same")(up6)
	#conv6 = tf.keras.layers.Conv2D(filters=dims*8, kernel_size=3, activation=lrelu, padding="same")(conv6)

	up7 = upsample_and_concat(conv6, conv3, dims*4, dims*8, is_fine=True, block_idx=1)
	conv7 = tf.keras.layers.Conv2D(filters=dims*4, kernel_size=3, activation='relu', padding="same")(up7)
	#conv7 = tf.keras.layers.Conv2D(filters=dims*4, kernel_size=3, activation=lrelu, padding="same")(conv7)

	up8 = upsample_and_concat(conv7, conv2, dims*2, dims*4, is_fine=True, block_idx=2)
	conv8 = tf.keras.layers.Conv2D(filters=dims*2, kernel_size=3, activation='relu', padding="same")(up8)
	#conv8 = tf.keras.layers.Conv2D(filters=dims*2, kernel_size=3, activation=lrelu, padding="same")(conv8)

	up9 = upsample_and_concat(conv8, conv1, dims, dims*2, is_fine=True, block_idx=3)
	conv9 = tf.keras.layers.Conv2D(filters=dims, kernel_size=3, activation=lrelu, padding="same")(up9)

	# if demosaic == True:
	# 	conv10 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, activation=None, padding="same")(conv9)
	# 	out = tf.nn.depth_to_space(input=conv10, block_size=int(np.sqrt(out_channels/3)))
	# else:
	out = tf.keras.layers.Conv2D(filters=dims, kernel_size=1, activation=None, padding="same")(conv9)
	#out = tf.keras.layers.Conv2D(filters=dims, kernel_size=1, activation=None, padding="same")(out)

	pred_y = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=(2, 2), padding='same', activation=None)(
		out)
	pred_uv = tf.keras.layers.Conv2D(filters=2, kernel_size=5, strides=(1, 1), padding='same', activation=None)(out)
	pred_y = pred_y + inp_y1
	pred_uv = pred_uv + inp_uv1
	return pred_y,pred_uv


def se_block(x, dims, block_idx, ratio=16):
	squeeze = tf.reduce_mean(input_tensor=x, axis=[1,2]) ## global avg pool.
	excitation = tf.keras.layers.Dense(units=dims//ratio)(squeeze)
	excitation = tf.nn.relu(excitation)
	excitation = tf.keras.layers.Dense(units=dims)(excitation)
	excitation = tf.sigmoid(excitation)
	excitation = tf.reshape(excitation, [-1,1,1,dims])

	#out = x*(0.5+excitation)
	out = x*excitation
	return out
