
import sys
sys.path.append("..")

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from skimage.transform import resize

def DepthNorm(x, maxDepth):
    return maxDepth / x

def scale_up(scale, images):
    scaled = []

    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append( resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True ) )

    return np.stack(scaled)

def load_depth_net_model():
    sess_depth_net = tf.Session()
    init_op_f = tf.global_variables_initializer()
    sess_depth_net.run(init_op_f)

    # load the depth model
    f = gfile.FastGFile("/ssd_scratch/shankara/VS_obs/RTVS/DenseDepth/depth_net/tf_model.pb", 'rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    f.close()
    sess_depth_net.graph.as_default()
    tf.import_graph_def(graph_def)

    # depth computation graph
    input_image = sess_depth_net.graph.get_tensor_by_name('import/input_1:0')
    pred_depth_tensor = sess_depth_net.graph.get_tensor_by_name('import/conv3/BiasAdd:0')
    return input_image, pred_depth_tensor, sess_depth_net

