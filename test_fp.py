import math
import time

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

fp_ops_module = tf.load_op_library("lib/libfp_par_2d.so")
projector = fp_ops_module.forward_projection_parallel2d
proj_grad = fp_ops_module.forward_projection_parallel2d_grad

@ops.RegisterGradient("ForwardProjectionParallel2D")
def _forward_projection_parallel2d_grad(op, grad):
    img_shape = op.get_attr("img_shape")
    img_space = op.get_attr("img_space")
    img_offset = op.get_attr("img_offset")
    proj_shape = op.get_attr("proj_shape")
    channel_space = op.get_attr("channel_space")
    channel_offset = op.get_attr("channel_offset")
    orbit_start = op.get_attr("orbit_start")
    orbit = op.get_attr("orbit")
    fov = op.get_attr("fov")
    method = op.get_attr("method")
    in_grad = proj_grad(proj=grad, img_shape=img_shape, img_space=img_space, 
            img_offset=img_offset, proj_shape=proj_shape, channel_space=channel_space, 
            channel_offset=channel_offset, orbit_start=orbit_start, orbit=orbit, 
            fov=fov, method=method)
    return [in_grad]

img1 = np.fromfile("phantom.raw", dtype=np.double)
img1.shape = 1, 512, 512, 1
img = np.zeros([4, 512, 512, 1], dtype=np.double)
img[0,:,:,:] = img1
img[1,:,:,:] = img1[:,::-1,:,:]
img[2,:,:,:] = img1[:,:,::-1,:]
img[3,:,:,:] = img1[:,::-1,::-1,:]

with tf.device('/gpu:1'):
    img_placeholder = tf.placeholder(tf.float64, [None, 512, 512, 1])
    proj = projector(img=img_placeholder, img_shape=[512, 512], img_space=[0.1, 0.1], 
            img_offset=[0, 0], proj_shape=[360, 860], channel_space=0.1, 
            channel_offset=0, orbit_start=0,orbit=math.pi/180, fov=50, 
            method="raydriven")
    proj_placeholder = tf.placeholder(tf.float64, [None, 360, 860, 1])
    gradient = proj_grad(proj=proj_placeholder, img_shape=[512, 512], img_space=[0.1, 0.1], 
            img_offset=[0, 0], proj_shape=[360, 860], channel_space=0.1, channel_offset=0, 
            orbit_start=0, orbit=math.pi/180, fov=50, method="raydriven")
    train_img = tf.get_variable('train_img', dtype=tf.float64, initializer=np.zeros([4, 512, 512, 1]))
    proj_train = projector(img=train_img, img_shape=[512, 512], img_space=[0.1, 0.1], 
            img_offset=[0, 0], proj_shape=[360, 860], channel_space=0.1, 
            channel_offset=0, orbit_start=0, orbit=math.pi/180, fov=50, 
            method="raydriven")
    # loss = tf.losses.mean_squared_error(proj_train, proj_placeholder)
    loss = tf.reduce_sum(tf.square(proj_train-proj_placeholder))
    dif = tf.losses.mean_squared_error(train_img, img_placeholder)
    adam_train = tf.train.AdamOptimizer(0.1).minimize(loss)
    sgd_train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
begin = time.time()
proj_img = sess.run(proj, feed_dict={img_placeholder: img})
duration = time.time() - begin
print("Cost %f seconds", duration)
"""
begin = time.time()
grad = sess.run(gradient, feed_dict={proj_placeholder: proj_img})
duration = time.time() - begin
print("Cost %f seconds", duration)
for i in range(10):
    proj_img /= 2
    begin = time.time()
    grad = sess.run(gradient, feed_dict={proj_placeholder: proj_img})
    duration = time.time() - begin
    print("Cost %f seconds", duration)
print(grad.shape)
print(np.max(grad, axis=(1, 2, 3)), np.min(grad, axis=(1, 2, 3)))

grad.tofile("gradient.raw")"""

# train image to fit projection
# for i in range(100):
#     _ = sess.run([adam_train], feed_dict={img_placeholder: img, proj_placeholder: proj_img})
for i in range(2000):
    begin = time.time()
    _, timg, mse_img, mse_proj = sess.run([sgd_train, train_img, dif, loss], 
            feed_dict={img_placeholder: img, proj_placeholder: proj_img})
    duration = time.time() - begin
    print("Iteration %d: img mse=%f, proj mse=%f, cost %f seconds" % (i, mse_img, mse_proj,
        duration))

timg.tofile("trained_img.raw")
