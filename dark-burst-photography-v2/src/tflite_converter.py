import tensorflow as tf
# import burst_nets
# import burst_nets_concatenate
# import burst_nets_no_map_fn_no_norm_yuv
import burst_nets_no_map_fn_no_norm_yuv_no_std
# Convert the model
method_name = "burst_l1_res_se_motion_cx"
saved_model_dir = '../saved_model_v2_light_4_res_2_encoder_1_no_map_fn_no_norm/YUV_-20_0_4_mean_2k_no_batch_1_25_y0uv0_cat/%s/'% method_name
model = tf.keras.models.load_model(saved_model_dir, compile=False)
# model=tf.keras.models.load_model('./ED-1536-2048', compile=False)
# model=tf.keras.models.load_model('./ED-720-1280', compile=False)
# model=tf.keras.models.load_model('./ED-3000-4000', compile=False)
# print(model.summary())
# exit()
# model=tf.keras.models.load_model('./ED-720-1280', compile=False)
model.save_weights('../checkpoints_v2_weights/my_checkpoint')

# load checkpoint
# model = Encoder_Decoder(self.h, self.w, self.chns, self.n_levels, self.scale, self.batch_size)
# model = UIC_NET(y_h=1536, y_w=2048, uv_h=768, uv_w=1024).get_UIC_NET()
# model = UIC_NET(y_h=768, y_w=1280, uv_h=384, uv_w=640).get_UIC_NET()
# model = UIC_NET(y_h=1728, y_w=2304, uv_h=864, uv_w=1152).get_UIC_NET()
# model = UIC_NET(y_h=1728, y_w=3840, uv_h=864, uv_w=1920).get_UIC_NET()
# in_image = tf.keras.Input(shape=[2048, 1536, 6], dtype=tf.float32, batch_size=3)
in_image_y1 = tf.keras.Input(shape=[4096, 3072, 1], dtype=tf.float32, batch_size=1)
in_image_uv1 = tf.keras.Input(shape=[2048, 1536, 2], dtype=tf.float32, batch_size=1)
in_image_y2 = tf.keras.Input(shape=[2048, 1536, 1], dtype=tf.float32, batch_size=1)
in_image_uv2 = tf.keras.Input(shape=[1024, 768, 2], dtype=tf.float32, batch_size=1)
in_image_y3 =tf.keras.Input(shape=[1024, 768, 2], dtype=tf.float32, batch_size=1)
in_image_uv3 = tf.keras.Input(shape=[512, 384, 4], dtype=tf.float32, batch_size=1)
# in_image_y = tf.keras.Input(shape=[16, 12, 1], dtype=tf.float32, batch_size=1)
# in_image_uv = tf.keras.Input(shape=[8, 6, 2], dtype=tf.float32, batch_size=1)
# out_image = burst_nets_no_map_fn_no_norm_yuv.fine_res_net_light(in_image)
out_image = burst_nets_no_map_fn_no_norm_yuv_no_std.fine_mimo(in_image_y1,in_image_uv1,in_image_y2,in_image_uv2,in_image_y3,in_image_uv3)
# model2 =tf.keras.Model(inputs=[in_image], outputs=[out_image],name='dbp')
model2 =tf.keras.Model(inputs=[in_image_y1,in_image_uv1,in_image_y2,in_image_uv2,in_image_y3,in_image_uv3], outputs=[out_image],name='dbp')
model2.summary()
# model = UIC_NET(y_h=1500, y_w=2000, uv_h=750, uv_w=1000).get_UIC_NET()
# model = UIC_NET(y_h=768, y_w=1280, uv_h=384, uv_w=640).get_UIC_NET()
# model.build(input_shape=[(1, 720, 1280, 1), (1, 360, 640, 2)])
model2.load_weights('../checkpoints_v2_weights/my_checkpoint').expect_partial()
model2.save(saved_model_dir+"tflite")

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir+"tflite") # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('saved_model_v2_light_4_res_2_encoder_1_no_map_fn_no_norm_yuv_std_no_batch_light_1_25_y0uv0_cat.tflite', 'wb') as f:
  f.write(tflite_model)
