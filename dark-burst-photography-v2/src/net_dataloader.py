from glob import glob
import random
import tensorflow as tf
import os
import pathlib
import numpy as np
import pandas as pd
import cv2
DEBUG = 1

def invnormalize(input_image):
    return ((input_image + 1.0) / (2.0 / 255.0))

class Net_DataLoader(object):

    def __init__(self,train_path, valPath, crop_size_h = -1,crop_size_w=-1, batch_size = 5, max_pixel_val = 255.0,
                 prefetch = 2,normalize=True,
                 strategy=None):

        self.train_path = train_path
        self.valPath = valPath
        self.batch_size = batch_size
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.max_pixel_val = max_pixel_val
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.prefetch = prefetch
        self.normalize_flag = normalize
        # self.min_ssim = min_ssim
        # self.max_ssim = max_ssim
        # self.sharpness_factor = sharpness_factor
        self.strategy = strategy
        self.resize = True


    def get_file_paths(self, folder_path):
        image_paths = []
        #get image files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.png',".jpg",".jpeg")):
                    image_paths.append(root + "/" + file)

        return image_paths

    def get_json_paths(self,folder_path):
        folder_paths = []
        #get scene folders
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith((".json")):
                    folder_paths.append(root + "/")

        return folder_paths


    def RGB2YUV(self, rgb):
        m = tf.constant([[ 0.29900, -0.16874,  0.50000],
                     [0.58700, -0.33126, -0.41869],
                     [ 0.11400, 0.50000, -0.08131]])
        yuv = tf.tensordot(rgb, m, axes = 1)
        y = yuv[:,:,0]
        u = tf.math.add(yuv[:,:,1],tf.constant(128.0, dtype=tf.float32))
        v = tf.math.add(yuv[:,:,2],tf.constant(128.0, dtype=tf.float32))
        newYUV = tf.stack([y, u, v], axis=2)
        newYUV = tf.clip_by_value(newYUV,0.0,255.0)
        return newYUV

    def YUV444toYUV420(self, yuv444):
        y = yuv444[:,:,0]
        uv = yuv444[:,:,1:]
        uv = (uv[0::2,0::2,:] + uv[0::2,1::2,:] + uv[1::2,0::2,:] + uv[1::2,1::2,:]) * 0.25
        return y, uv

    def normalize(self, input_image):
        return (input_image / (self.max_pixel_val/2.0)) - 1


    def random_crop(self, gt_image,input_images):
        if(gt_image.shape != input_images[0].shape):
            gt_image = tf.image.resize_with_crop_or_pad(gt_image,target_height=3072,target_width=4080)
        images = [gt_image] + input_images
        stacked_image = tf.stack(images, axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[len(images), self.crop_size_h, self.crop_size_w, 3])
        return cropped_image[0], cropped_image[1:]
 

    def get_files_from_folder_test(self, folder_path, input_evs):
        files = os.listdir(folder_path)
        files.sort()
        jpgs = []
        for f in files:
            if (b'.jpg' in f):
                jpgs.append(f)

        evs = {}
        for jpg in jpgs:
            if (b'_ev_' in jpg):
                index = jpg.index(b'_ev_')
                ev_num = jpg[index + 4:-4]
                ev_num = float(ev_num)
                if (ev_num not in evs.keys()):
                    evs[ev_num] = []
                    evs[ev_num].append(jpg)
                else:
                    evs[ev_num].append(jpg)
        inp_ev_paths = []
        for inp_ev, num_frames in input_evs:

            inp_ev_paths.extend(evs[inp_ev][0:num_frames])

        inp_ev_paths = [folder_path + i for i in inp_ev_paths]

        return  inp_ev_paths


    def get_files_from_folder(self,folder_path, input_evs):
        files = os.listdir(folder_path)
        files.sort()
        jpgs = []
        for f in files:
            if (b'.jpg' in f):
                jpgs.append(f)

        gts = []
        evs = {}
        for jpg in jpgs:
            if (b'_gt_' in jpg):
                gts.append(jpg)
            elif (b'_ev_' in jpg):
                index = jpg.index(b'_ev_')
                ev_num = jpg[index + 4:-4]
                ev_num = float(ev_num)
                if (ev_num not in evs.keys()):
                    evs[ev_num] = []
                    evs[ev_num].append(jpg)
                else:
                    evs[ev_num].append(jpg)
        gts.sort()
        gt_path = folder_path + gts[0]
        inp_ev_paths = []
        for inp_ev, num_frames in input_evs:
            inp_ev_paths.extend(evs[inp_ev][0:num_frames])

        inp_ev_paths = [folder_path + i for i in inp_ev_paths]

        return gt_path, inp_ev_paths


    def anv_trainDataLoader(self,input_evs):
        folder_paths = self.get_json_paths(self.train_path)

        train_dataset = tf.data.Dataset.from_generator(generator=self.anv_generator,args=[folder_paths,input_evs,self.batch_size], output_types=(tf.float32,tf.float32,
                                                                                                  tf.float32,tf.float32),
                                                       output_shapes=((None,self.crop_size_h,self.crop_size_w,1),(None,self.crop_size_h//2,self.crop_size_w//2,2),
                                                                      (None,self.crop_size_h,self.crop_size_w,3),(None,self.crop_size_h//2,self.crop_size_w//2,6)))
        
        if(self.strategy):
            train_dataset = train_dataset.batch(self.batch_size * self.strategy.num_replicas_in_sync).prefetch(self.prefetch)
            train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        else:
            train_dataset = train_dataset.prefetch(self.prefetch)

        return train_dataset


    def anv_generator(self,folder_paths, input_evs,batch_size):
        DEBUG = 1
        i=0
        np.random.shuffle(folder_paths)
        while True:
            if i*batch_size >= len(folder_paths):
                i=0
                return
            else:
                batch_chunk = folder_paths[i*batch_size:(i+1)*batch_size]
                target_batch_y = []
                target_batch_uv = []
                input_batch_y = []
                input_batch_uv = []
                for path in batch_chunk:
                    gt_file_path, inp_ev_paths = self.get_files_from_folder(path, input_evs=input_evs)
                    # extract images
                    target_image = tf.io.read_file(gt_file_path)
                    target_image = tf.image.decode_image(target_image, channels=3)

                    input_images = []
                    for inp_ev_path in inp_ev_paths:
                        input_image = tf.io.read_file(inp_ev_path)
                        input_image = tf.image.decode_image(input_image, channels=3)
                        input_images.append(input_image)

                    # random crop
                    if (self.crop_size_h != -1):
                        target_image, input_images = self.random_crop(target_image, input_images)

                    # TODO convert to mat-math
                    # Convert to YUV444 and then to YUV420
                    input_images_y420 = []
                    input_images_uv420 = []
                    for input_image in input_images:
                        input_image_444 = self.RGB2YUV(tf.cast(input_image, tf.float32))
                        input_image_y420, input_image_uv420 = self.YUV444toYUV420(input_image_444)
                        if (self.normalize_flag):
                            input_image_y420, input_image_uv420 = self.normalize(input_image_y420), self.normalize(
                                input_image_uv420)
                        input_image_y420 = tf.expand_dims(input_image_y420, axis=2)

                        input_images_y420.append(input_image_y420)
                        # print("input_image_y420.shape",input_image_y420.shape)
                        input_images_uv420.append(input_image_uv420)

                    #concat input images along channels
                    input_images_y420 = np.concatenate(np.asarray(input_images_y420),axis=-1)
                    input_images_uv420 = np.concatenate(np.asarray(input_images_uv420),axis=-1)

                    target_image_444 = self.RGB2YUV(tf.cast(target_image, tf.float32))
                    target_image_y420, target_image_uv420 = self.YUV444toYUV420(target_image_444)
                    if (self.normalize_flag):
                        target_image_y420, target_image_uv420 = self.normalize(target_image_y420), self.normalize(
                            target_image_uv420)
                    target_image_y420 = tf.expand_dims(target_image_y420,axis=2)
                    target_batch_y.append(target_image_y420)
                    target_batch_uv.append(target_image_uv420)
                    input_batch_y.append(input_images_y420)
                    input_batch_uv.append(input_images_uv420)


                target_batch_y = np.asarray(target_batch_y)#.reshape((-1,1024,1024,1))
                target_batch_uv = np.asarray(target_batch_uv)#.reshape((-1,512,512,2))
                input_batch_y = np.asarray(input_batch_y)#.reshape((-1,3,-1,-1,-1))
                input_batch_uv = np.asarray(input_batch_uv)#.reshape((-1,3,-1,-1,-1))

                yield target_batch_y, target_batch_uv , input_batch_y, input_batch_uv
                i += 1
