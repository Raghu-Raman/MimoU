# generate graph
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

input_arrays = ["Placeholder"]
output_arrays = ["g_conv10_fine/BiasAdd"]
graph_def_file = '../checkpoint_original/Sony/burst_l1_res_se_motion_cx/frozen_model.pb'
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    '../checkpoint_original/Sony/burst_l1_res_se_motion_cx/frozen_model.pb', input_arrays=input_arrays,
    output_arrays=output_arrays, input_shapes={"Placeholder": [None, 1, 1424, 2128, 4]})

converter.experimental_new_converter = True
tflite_model = converter.convert()
with open('../checkpoint_original/Sony/burst_l1_res_se_motion_cx/model.tflite', 'wb') as f:
    f.write(tflite_model)
    print('done')
    exit(0)