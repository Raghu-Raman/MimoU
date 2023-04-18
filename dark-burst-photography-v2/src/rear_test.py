# test.py
# description: Evaluation script for the set based model.

from __future__ import division
import os, scipy.io, time
import tensorflow as tf
import numpy as np
import rawpy
import glob
import dbputils
import cv2
from net_dataloader import *
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


n_burst = 9
#input_dir = '../dataset/YUV/short_jpg/'
#input_dir = '../dataset/DSLR/short/'
#input_dir = '../dataset/burst_11_test/short/'
#input_dir="../dataset/Competition_testing_10_10_jpg_only/short_jpg/"
method_name = "burst_l1_res_se_motion_cx"
#gt_dir = '../dataset/YUV/long_jpg/'
#gt_dir = '../dataset/DSLR/long/'
#gt_dir = '../dataset/burst_11_test/long/'
#gt_dir = '../dataset/Competition_testing_10_10_jpg_only/long_jpg/'
method = "burst_l1_res_se_motion_cx"
d_id = 1 ## 1 for test, 2 for validation
result_name = "burst_l1_res_se_motion_cx"
d_set = dbputils.d_set_for_id(d_id)
checkpoint_dir = '../checkpoint/Sony/%s/'%method
result_dir = '../results_test_rear/'
gt_result_dir = '../results_one_step_v2_light_12_res_2_encoder_2_no_map_fn_no_norm_yuv/%s/ground_truth/'%(d_set)
#saved_model_dir='../saved_model_v2_light_4_res_2_encoder_1_no_map_fn_no_norm/YUV_-20_0_4_mean_2k_no_batch_11_30/%s/'% method_name
saved_model_dir='../saved_model_v2_light_4_res_2_encoder_1_no_map_fn_no_norm/YUV_-20_0_4_mean_2k_no_batch_1_25_y0uv0_cat/%s/'% method_name

if not os.path.isdir(result_dir):
	os.makedirs(result_dir)
if not os.path.isdir(gt_result_dir):
        os.makedirs(gt_result_dir)
is_burst = True
def G_loss(out_image, gt_image):
	G_l1 = tf.reduce_mean(input_tensor=tf.abs(gt_image - out_image))
	G_pix = G_l1

	# vgg_fake = build_vgg19(255*out_image)
	# vgg_real = build_vgg19(255*gt_image, reuse=True)
	#
	# # # ## CX Loss.
	# CX_loss_list = [CX_loss_helper(vgg_real[layer], vgg_fake[layer], config.CX)
	# 						for layer, w in config.CX.feat_layers.items()]
	# CX_loss = tf.reduce_sum(input_tensor=CX_loss_list)
	# G_feat = CX_loss/255.

	# ## Perceptual loss.
	# per_loss_list = [tf.reduce_mean(tf.abs(vgg_real[layer]-vgg_fake[layer])) for layer, w in config.CX.feat_layers.items()]
	# per_loss = tf.reduce_mean(per_loss_list)/255.
	# G_feat = 0.1*per_loss

	#G_loss_hd = G_pix + G_feat
	G_loss_hd = G_pix
	G_loss = G_loss_hd

	return G_loss
# get test IDs

v2_model = tf.keras.models.load_model(saved_model_dir,custom_objects={'G_loss':G_loss}, compile=False)
v2_model.summary()
v2_model.save_weights('../checkpoints_v2_weights/my_checkpoint')

ssim_list = []
psnr_list = []
lpips_list = []
time_list = []
ratio_list = []
count = 0


def space_to_depth(x, block_size=2):
	x = np.asarray(x)
	print("x shape", x.shape)
	batch = x.shape[0]
	height = x.shape[1]
	width = x.shape[2]

	depth = x.shape[3]
	reduced_height = height // block_size
	reduced_width = width // block_size
	y = x.reshape(batch, reduced_height, block_size,
				  reduced_width, block_size, depth)
	z = np.swapaxes(y, 2, 3).reshape(batch, reduced_height, reduced_width, -1)
	return z
def depth_to_space(x, block_size=2):
	x = np.asarray(x)
	height = x.shape[0]
	width = x.shape[1]

	depth = x.shape[2]

	original_height = height * block_size
	original_width = width * block_size
	original_depth=depth//(block_size**2)
	x = x.reshape(height, width, block_size,block_size)
	x= x.transpose([0,2,1,3]).reshape(original_height,original_width)
	#x=np.swapaxes(x,1,2).reshape(original_height,original_width)
	x=np.expand_dims(x,axis=2)

	return x

def YUV444toYUV420(yuv444):

    y = yuv444[:,:,0]
    y = tf.expand_dims(y, axis=2)
    uv = yuv444[:,:,1:]
    uv=np.float32(uv)
    #print((uv[0::2,0::2,:] + uv[0::2,1::2,:] + uv[1::2,0::2,:] + uv[1::2,1::2,:]))
    uv = (uv[0::2,0::2,:] + uv[0::2,1::2,:] + uv[1::2,0::2,:] + uv[1::2,1::2,:]) / 4.0
    return y, np.uint8(uv)

def YUV420toYUV444(y, uv):  # format NHWC

    H2 = np.shape(uv)[0]
    W2 = np.shape(uv)[1]
    #
    tmpuv = np.concatenate([uv, uv], axis = 2)


    tmpuv = np.reshape(tmpuv, [H2,2*W2,2])


    tmpuv = np.concatenate([tmpuv, tmpuv], axis = 1)


    tmpuv = np.reshape(tmpuv, [2*H2,2*W2,2])

    #tmpuv=depth_to_space(np.concatenate([uv, uv, uv, uv], axis = 3),2)
    # tmpuv[0::2, 0::2, :]=uv
    # tmpuv[0::2, 1::2 ,:]=uv
    # tmpuv[1::2, 0::2 ,:]=uv
    # tmpuv[1::2, 1::2 ,:]=uv
    yuv444 = np.concatenate([y, tmpuv], axis = 2)
    yuv444 = np.clip(yuv444, 0, 255)
    # if DEBUG: print(np.shape(yuv444))

    return yuv444

def YUV2RGB(yuv):
    m = np.array([[1.0, 1.0, 1.0],
                  [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                  [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])

    rgb = np.dot(yuv, m)
    rgb[:, :, 0] -= 179.45477266423404
    rgb[:, :, 1] += 135.45870971679688
    rgb[:, :, 2] -= 226.8183044444304

    rgb = np.clip(rgb, 0, 255)
    return rgb



def YUV420toRGB(y,uv):
    yuv444 = YUV420toYUV444(y,uv)

    # rgbs=[]
    # for yuv in yuv444:
    #     rgb = YUV2RGB(yuv)
    #     rgb=tf.multiply(rgb,1.0/255)
    #     rgbs.append(rgb)
    rgb = YUV2RGB(yuv444)
    rgb = np.multiply(rgb, 1.0 / 255)
    # rgbs.append(rgb)

    # rgbs = tf.stack(rgbs,)
    return rgb
# dataloader = Net_DataLoader(train_path="../dataset/Competition_testing_10_10_jpg_only/",
# 							valPath="../dataset/Competition_testing_10_10_jpg_only/", crop_size_h=-1,
# 							crop_size_w=-1,
# 							batch_size=1, normalize=True)

dataloader = Net_DataLoader(train_path="../../../dataset/rear_dataset/test_rear/",
							valPath="../../../dataset/rear_dataset/test_rear", crop_size_h=-1,
							crop_size_w=-1,
							batch_size=1, normalize=True)
# dataloader = Net_DataLoader(train_path="../dataset/Combine/",
# 							valPath="../dataset/Combine/", crop_size_h=-1,
# 							crop_size_w=-1,
# 							batch_size=1, normalize=True)
#test_loader = dataloader.anv_trainDataLoader(input_evs=[(-20, 1), (0, 1), (4, 1)])
#folder_paths = dataloader.get_json_paths("../dataset/test_11/")
test_loader = dataloader.anv_testDataLoader(input_evs=[(-20,1),(0,1),(4,1)])
test_id=0
for data in test_loader:
	
	parent_dir = "../results_test_rear/"

	mode = 0o777
	y_temp = data[0]
	uv_temp = data[1]
	print("y_temp.shape",y_temp.shape)

	uv_temp = np.float32(uv_temp)
	uv_temp_patches = np.minimum(uv_temp, 1.0)
	y_temp = np.float32(y_temp)
	y_temp_patches = np.minimum(y_temp, 1.0)
	st = time.time()
	y_temp_patches1 = y_temp_patches[:,:,:,0:1]
	uv_temp_patches1 = uv_temp_patches[:,:,:,0:2]

	y_temp_patches2 = y_temp_patches[:,:,:,1:2]
	uv_temp_patches2 = uv_temp_patches[:,:,:,2:4]
	y_temp_patches2 = tf.image.resize(y_temp_patches2,[y_temp_patches2.shape[1]//2,y_temp_patches2.shape[2]//2])
	uv_temp_patches2 = tf.image.resize(uv_temp_patches2,[uv_temp_patches2.shape[1]//2,uv_temp_patches2.shape[2]//2])
				
	y_temp_patches3 = y_temp_patches[:,:,:,1:3]
	uv_temp_patches3 = uv_temp_patches[:,:,:,2:6]
	y_temp_patches3 = tf.image.resize(y_temp_patches3,[y_temp_patches3.shape[1]//4,y_temp_patches3.shape[2]//4])
	uv_temp_patches3 = tf.image.resize(uv_temp_patches3,[uv_temp_patches3.shape[1]//4,uv_temp_patches3.shape[2]//4])
	
	output_y,output_uv = v2_model([y_temp_patches1,uv_temp_patches1,y_temp_patches2,uv_temp_patches2,y_temp_patches3,uv_temp_patches3], training=False)
	ev_0_y = data[0][:,:,:,1:2]
	ev_0_uv = data[1][:,:,:,2:4]
	ev_m20_y = data[0][:,:,:,0:1]
	ev_m20_uv = data[1][:,:,:,0:2]
	
	output_y = np.minimum(output_y, 1)
	output_uv = np.minimum(output_uv, 1)
	ev_0_y = np.minimum(ev_0_y, 1)
	ev_0_uv =  np.minimum(ev_0_uv, 1)
	ev_m20_y = np.minimum(ev_m20_y, 1)
	ev_m20_uv =  np.minimum(ev_m20_uv, 1)
	
	time_ = time.time() - st
	y_output = output_y[0, :, :, :]
	uv_output = output_uv[0, :, :, :]
	ev_0y_output = ev_0_y[0, :, :, :]
	ev_0uv_output = ev_0_uv[0, :, :, :]
	ev_m20y_output = ev_m20_y[0, :, :, :]
	ev_m20uv_output = ev_m20_uv[0, :, :, :]
	output = YUV420toYUV444((y_output + 1) * 255.0 / 2.0, (uv_output + 1) * 255.0 / 2.0)
	ev0 = YUV420toYUV444((ev_0y_output + 1) * 255.0 / 2.0, (ev_0uv_output + 1) * 255.0 / 2.0)
	evm20 = YUV420toYUV444((ev_m20y_output + 1) * 255.0 / 2.0, (ev_m20uv_output + 1) * 255.0 / 2.0)
	# y_output = (output[:, :, 0:4] +1)*255.0/2.0
	# uv_output = (output[:, :, 4:6]+1)*255.0/2.0
	# print("y_output shape", y_output.shape)
	# print("uv_output shape", uv_output.shape)
	# y_output = depth_to_space(y_output, 2)
	#
	# output = YUV420toYUV444(y_output, uv_output)
	#
	# print("output shape,", output.shape)
	# print("gt_full shape,", gt_full.shape)
	# gt_full_test = YUV2RGB(gt_full) / 255
	# output_test = YUV2RGB(output) / 255
	# output_test = np.minimum(np.maximum(output_test, 0), 1)
	# gt_full_test = np.minimum(np.maximum(gt_full_test, 0), 1)
	# print(output_test)
	# print(gt_full_test)
	# ssim_val = structural_similarity(output_test, gt_full_test, multichannel=True)
	# psnr_val = peak_signal_noise_ratio(output_test, gt_full_test)
	# count += 1
	# ssim_list.append(ssim_val)
	# psnr_list.append(psnr_val)
	# time_list.append(time_)
	#
	# avg_ssim = np.mean(ssim_list)
	# avg_psnr = np.mean(psnr_list)
	# avg_lpips = np.mean(lpips_list)
	# avg_time = np.mean(time_list[1:])
	#
	# # print("ssim: %.4f, psnr: %.4f, lpips: %.4f, Time elapsed: %.3f, %d, %d" % (ssim_val, psnr_val, lpips_val, time_, ratio, count))
	# print("Avg ssim: %.4f, Avg psnr: %.4f, Avg lpips: %.4f, Time elapsed: %.3f,avg time: %.3f, %d" % (
	# 	avg_ssim, avg_psnr, avg_lpips, time_, avg_time, count))

	print("test time:", time_)
	## Save output images.
	# output = np.minimum(np.maximum(output / 255, 0), 1)

	temp = cv2.cvtColor(np.uint8(output), cv2.COLOR_YUV2BGR)
	temp_ev0  = cv2.cvtColor(np.uint8(ev0), cv2.COLOR_YUV2BGR)
	temp_evm20  = cv2.cvtColor(np.uint8(evm20), cv2.COLOR_YUV2BGR)
	# print(tf.reduce_sum(temp))
	directory =str(test_id) + "_Output_folder"
	path = os.path.join(parent_dir, directory)
	if not os.path.exists(path):
		os.mkdir(path, mode)
	print(path)
	resultPath = str(path) + "/"
	print(resultPath)
    
	cv2.imwrite(resultPath + "%05d_00_out.jpg" % (test_id), temp)
	cv2.imwrite(resultPath + "ev0.jpg", temp_ev0)
	cv2.imwrite(resultPath + "evm20.jpg", temp_evm20)
	
	# gt_full = np.minimum(np.maximum(gt_full, 0), 1)

	# print(gt_result_dir + "%05d_00_gt.jpg" % (test_id))

	# temp = cv2.cvtColor(np.uint8(gt_full), cv2.COLOR_YUV2BGR)
	# temp=np.uint8(gt_full * 255)
	# temp = cv2.cvtColor(np.uint8(YUV2RGB(gt_full)), cv2.COLOR_RGB2BGR)
	# print("the gt shape is", temp.shape)
	# cv2.imwrite(gt_result_dir + "%05d_00_gt.jpg" % (test_id), temp)
	test_id=test_id+1
	print("done.")


