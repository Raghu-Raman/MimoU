# train.py
# description: Training script for Burst Photography for Learning to Enhance Extremely Dark Images.

## Train coarse network with ps=256, 4000 epochs
## Train fine network with ps=512, ps_low=256, 4000 epochs without fixed params, lr=1e-4 -> 1e-5 after 2000 epochs.
## Train set-based fine network with ps_fine=512, ps_denoise=256, 1000 epochs with fixed coarse params.

from __future__ import division
import os, time
import tensorflow as tf
import numpy as np
import rawpy
from tensorflow import keras as K
import glob
import dbputils
import burst_nets_no_map_fn_no_norm_yuv_no_std
import cv2
from vgg import *
from CX.CX_helper import *
from net_dataloader import *
from CX.config import *

print("If eager?: ", tf.executing_eagerly())
method_name = "burst_l1_res_se_motion_cx"
checkpoint_dir = '../checkpoint/YUV_no_std_2k_bo_batch_1_25_y0uv0_cat/%s/' % method_name
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
result_dir = "../results/learning/%s" % method_name
saved_model_dir='../saved_model_v2_light_4_res_2_encoder_1_no_map_fn_no_norm/YUV_-20_0_4_mean_2k_no_batch_1_25_y0uv0_cat/%s/'% method_name
# get train IDs
dataloader = Net_DataLoader(train_path="../../../dataset/rear_dataset/panang_rear_dataset/",
							valPath="../../../dataset/rear_dataset/panang_rear_dataset/", crop_size_h=512,
							crop_size_w=512,
							batch_size=1, normalize=True)

folder_paths = dataloader.get_json_paths("../../../dataset/rear_dataset/panang_rear_dataset/")


save_freq = 1

train_coarse = False
n_burst = 9
max_burst = n_burst
ps = 256
gt_images = [None] * 6000
gt_image_raws = [None] * 6000
input_images = []
# g_loss = np.zeros((5000, 1))
g_loss_raw = np.zeros((5000, 1))
g_loss_hd = np.zeros((5000, 1))
g_loss_l1 = np.zeros((5000, 1))
g_loss_feat = np.zeros((5000, 1))

class DBP(object):
	def __init__(self):
		self.lr = 1e-4
		self.start_epoch=0
		self.num_epochs = 4001
		self.input_images=[]
		self.gt_images = [None] * 6000
		self.gt_image_raws = [None] * 6000
	@tf.function
	def G_loss(self, output_y,output_uv, y_temp_gt_patch,uv_temp_gt_patch):
		G_l1_y = tf.reduce_mean(input_tensor=tf.abs(output_y - y_temp_gt_patch))
		G_l1_uv=tf.reduce_mean(input_tensor=tf.abs(output_uv - uv_temp_gt_patch))
		# print("MS_SSIM",tf.image.ssim_multiscale(output_y,y_temp_gt_patch,filter_size=11,max_val=1.0,filter_sigma=1.5,k1=0.01,k2=0.03))
		# G_pix = G_l1_y+G_l1_uv
		alpha = 0.8
		l1_w = 1-alpha
		ms_ssim_w = alpha
    
		ms_ssim_y = tf.reduce_mean(1-tf.image.ssim_multiscale( output_y,y_temp_gt_patch, max_val = 1.0,filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03))
		ms_ssim_uv = tf.reduce_mean(1-tf.image.ssim_multiscale( output_uv,uv_temp_gt_patch, max_val = 1.0,filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03))
		loss = ms_ssim_w*(ms_ssim_y + ms_ssim_uv) + l1_w*(G_l1_y+G_l1_uv) 
		print(loss)
		return loss

		# vgg_fake = build_vgg19(255*out_image)
		# vgg_real = build_vgg19(255*gt_image, reuse=True)
		#
		# # ## CX Loss.
		# CX_loss_list = [CX_loss_helper(vgg_real[layer], vgg_fake[layer], config.CX)
		# 						for layer, w in config.CX.feat_layers.items()]
		# CX_loss = tf.reduce_sum(input_tensor=CX_loss_list)
		# G_feat = CX_loss/255.
		#
		# # Perceptual loss.
		# per_loss_list = [tf.reduce_mean(tf.abs(vgg_real[layer]-vgg_fake[layer])) for layer, w in config.CX.feat_layers.items()]
		# per_loss = tf.reduce_mean(per_loss_list)/255.
		# G_feat = 0.1*per_loss

		#G_loss_hd = G_pix + G_feat
		# G_loss_hd = G_pix
		# G_loss = G_loss_hd

		# return G_loss

## Training.
########################
	@tf.function
	def train_step(self, y_temp_patches1,uv_temp_patches1,y_temp_patches2,uv_temp_patches2,y_temp_patches3,uv_temp_patches3,y_temp_gt_patch,uv_temp_gt_patch):
		with tf.GradientTape() as ae_tape:
			#output = tf.minimum(tf.maximum(self.model(input_patches, training=True), 0), 1)
			output_y,output_uv = self.model([y_temp_patches1,uv_temp_patches1,y_temp_patches2,uv_temp_patches2,y_temp_patches3,uv_temp_patches3], training=True)
			# print("output_y",output_y.shape)
			# print("output_uv",output_uv.shape)
			loss = self.G_loss(output_y,output_uv, y_temp_gt_patch,uv_temp_gt_patch)
		ae_grads = ae_tape.gradient(loss, self.model.trainable_variables)
		# capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
		self.G_opt.apply_gradients(zip(ae_grads, self.model.trainable_variables))
		return loss

	def train(self):
		in_y1 = tf.keras.Input(shape=[ps*2, ps*2, 1], dtype=tf.float32, batch_size=1)
		in_uv1 = tf.keras.Input(shape=[ps, ps, 2], dtype=tf.float32, batch_size=1)
		in_y2 = tf.keras.Input(shape=[ps, ps, 1], dtype=tf.float32, batch_size=1)
		in_uv2 = tf.keras.Input(shape=[ps//2, ps//2, 2], dtype=tf.float32, batch_size=1)
		in_y3 = tf.keras.Input(shape=[ps//2, ps//2, 2], dtype=tf.float32, batch_size=1)
		in_uv3 = tf.keras.Input(shape=[ps//4, ps//4, 4], dtype=tf.float32, batch_size=1)
		gt_y = tf.keras.Input(shape=[ps * 2, ps * 2, 1], dtype=tf.float32, batch_size=1)
		gt_uv = tf.keras.Input(shape=[ps, ps, 2], dtype=tf.float32, batch_size=1)
		
		out_y,out_uv = burst_nets_no_map_fn_no_norm_yuv_no_std.fine_mimo(in_y1,in_uv1,in_y2,in_uv2,in_y3,in_uv3)

		self.model = tf.keras.Model(inputs=[in_y1,in_uv1,in_y2,in_uv2,in_y3,in_uv3], outputs=[out_y,out_uv], name='dbp')
		print("Starting training..")
		start_epoch = 0
		num_epochs = 500

		self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			initial_learning_rate=self.lr,
			decay_steps=150 * 2000,
			decay_rate=0.9)

		self.G_opt = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
		# self.G_opt=tf.keras.optimizers.SGD(
		# 	learning_rate=self.lr_schedule,
		# 	momentum=0.65,
		# 	nesterov=True,
		# 	name='SGD'
		# )

		self.model.compile(optimizer=self.G_opt, loss=self.G_loss)
		self.model.summary()
		tf.keras.utils.plot_model(self.model,to_file='model.png',show_shapes=True)	
		checkpoint = tf.train.Checkpoint(optimizer=self.G_opt, model=self.model)
		# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

		for epoch in range(start_epoch, num_epochs):
			cnt = 0
			print("Check the epoch and save frequency here",epoch%save_freq)
			if epoch % save_freq == 0 and epoch > 0:
				print("Saving model..")
				self.model.save(saved_model_dir)
				checkpoint.save(file_prefix=checkpoint_prefix)
			# r = np.random.randint(1,n_burst+1)
			#CHANGE THE EV HERE
			i=0
			errorList=0
			train_loader = dataloader.anv_trainDataLoader(input_evs=[(-24.0000000,1),(0,1),(4,1)])
			for data in train_loader:
				st = time.time()
				y_temp = data[2]
				#y_temp = y_temp[:, :, :, 0]
				uv_temp = data[3]
				# print("y_temp shape",y_temp.shape)
				# print("uv_temp shape",uv_temp.shape)
				#uv_temp = uv_temp[:, :, :, 0:]
				# y_temp=tf.transpose(y_temp, [3, 1, 2,0])
				# uv_temp=tf.concat((uv_temp_1,uv_temp_2,uv_temp_3),axis=0)
				# im = np.expand_dims(im, 0)
				uv_temp = np.float32(uv_temp)
				uv_temp_patches = np.minimum(uv_temp, 1.0)
				y_temp = np.float32(y_temp)
				y_temp_patches = np.minimum(y_temp, 1.0)
				## read gt.
				y_temp = data[0]
				uv_temp = data[1]
				# print("y_temp shape",y_temp.shape)
				# print("uv_temp shape",uv_temp.shape)
				y_temp_gt = np.float32(y_temp)
				y_temp_gt_patch = np.minimum(y_temp_gt, 1.0)

				uv_temp_gt = np.float32(uv_temp)
				uv_temp_gt_patch = np.minimum(uv_temp_gt, 1.0)

				y_temp_patches1 = y_temp_patches[:,:,:,0:1]
				uv_temp_patches1 = uv_temp_patches[:,:,:,0:2]
				# print("y_temp_patches1",y_temp_patches1.shape)
				# print("uv_temp_patches1",uv_temp_patches1.shape)
				

				y_temp_patches2 = y_temp_patches[:,:,:,1:2]
				uv_temp_patches2 = uv_temp_patches[:,:,:,2:4]
				y_temp_patches2 = tf.image.resize(y_temp_patches2,[y_temp_patches1.shape[1]//2,y_temp_patches1.shape[2]//2])
				uv_temp_patches2 = tf.image.resize(uv_temp_patches2,[uv_temp_patches1.shape[1]//2,uv_temp_patches1.shape[2]//2])
				# print("y_temp_patches2",y_temp_patches2.shape)
				# print("uv_temp_patches2",uv_temp_patches2.shape)
				
				y_temp_patches3 = y_temp_patches[:,:,:,1:3]
				uv_temp_patches3 = uv_temp_patches[:,:,:,2:6]
				y_temp_patches3 = tf.image.resize(y_temp_patches3,[y_temp_patches1.shape[1]//4,y_temp_patches1.shape[2]//4])
				uv_temp_patches3 = tf.image.resize(uv_temp_patches3,[uv_temp_patches1.shape[1]//4,uv_temp_patches1.shape[2]//4])
				# print("y_temp_patches3",y_temp_patches3.shape)
				# print("uv_temp_patches3",uv_temp_patches3.shape)
				
				
				st2 = time.time()
				loss = self.train_step(y_temp_patches1,uv_temp_patches1,y_temp_patches2,uv_temp_patches2,y_temp_patches3,uv_temp_patches3,y_temp_gt_patch,uv_temp_gt_patch)
				g_loss_hd[i] = loss


				print("%d %d Loss=%.4f Time=%.4f" % (epoch, cnt, np.mean(g_loss_hd[np.where(g_loss_hd)]), time.time() - st))
				cnt += 1

				i = i + 1
if __name__ == '__main__':

	dbp = DBP()
	dbp.train()

