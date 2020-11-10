# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:50:03 2019
@author: wmy
"""

import numpy as np
#import matplotlib.pyplot as plt
import pywt
import tensorflow as tf
#import pylab

#pylab.rcParams['figure.figsize'] = (10.0, 10.0)

def dwt2d(x, wave='haar'):
    # shape x: (b, h, w, c)
    nc = int(x.shape.dims[3])
    # 小波波形
    w = pywt.Wavelet(wave)
    # 水平低频 垂直低频
    ll = np.outer(w.dec_lo, w.dec_lo)
    # 水平低频 垂直高频
    lh = np.outer(w.dec_hi, w.dec_lo)
    # 水平高频 垂直低频 
    hl = np.outer(w.dec_lo, w.dec_hi)
    # 水平高频 垂直高频
    hh = np.outer(w.dec_hi, w.dec_hi)
    # 卷积核
    core = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
    core[:, :, 0, 0] = ll[::-1, ::-1]
    core[:, :, 0, 1] = lh[::-1, ::-1]
    core[:, :, 0, 2] = hl[::-1, ::-1]
    core[:, :, 0, 3] = hh[::-1, ::-1]
    core = core.astype(np.float32)
    kernel = np.array([core], dtype=np.float32)
    kernel = tf.convert_to_tensor(kernel)
    p = 2 * (len(w.dec_lo) // 2 - 1)
    with tf.compat.v1.variable_scope('dwt2d'):
        # padding odd length
        x = tf.pad(x, tf.constant([[0, 0], [p, p+1], [p, p+1], [0, 0]]))
        xh = tf.shape(x)[1] - tf.shape(x)[1]%2
        xw = tf.shape(x)[2] - tf.shape(x)[2]%2
        x = x[:, 0:xh, 0:xw, :]
        # convert to 3d data
        x3d = tf.expand_dims(x, 1)
        # 切开通道
        x3d = tf.split(x3d, int(x3d.shape.dims[4]), 4)
        # 贴到维度一
        x3d = tf.concat([a for a in x3d], 1)
        # 三维卷积
        y3d = tf.nn.conv3d(x3d, kernel, padding='VALID', strides=[1, 1, 2, 2, 1])
        # 切开维度一
        y = tf.split(y3d, int(y3d.shape.dims[1]), 1)
        # 贴到通道维
        y = tf.concat([a for a in y], 4)
        y = tf.reshape(y, (tf.shape(y)[0], tf.shape(y)[2], tf.shape(y)[3], 4*nc))
        # 拼贴通道
        channels = tf.split(y, nc, 3)
        outputs = []
        for channel in channels:
            (cA, cH, cV, cD) = tf.split(channel, 4, 3)
            AH = tf.concat([cA, cH], axis=2)
            VD = tf.concat([cV, cD], axis=2)
            outputs.append(tf.concat([AH, VD], axis=1))
            pass
        outputs = tf.concat(outputs, axis=-1)
        pass
    return outputs

def wavedec2d(x, level=1, wave='haar'):
    if level == 0:
        return x
    y = dwt2d(x, wave=wave)
    hcA = tf.math.floordiv(tf.shape(y)[1], 2)
    wcA = tf.math.floordiv(tf.shape(y)[2], 2)
    cA = y[:, 0:hcA, 0:wcA, :]
    cA = wavedec2d(cA, level=level-1, wave=wave)
    cA = cA[:, 0:hcA, 0:wcA, :]
    hcA = tf.shape(cA)[1]
    wcA = tf.shape(cA)[2]
    cH = y[:, 0:hcA, wcA:, :]
    cV = y[:, hcA:, 0:wcA, :]
    cD = y[:, hcA:, wcA:, :]
    AH = tf.concat([cA, cH], axis=2)
    VD = tf.concat([cV, cD], axis=2)
    outputs = tf.concat([AH, VD], axis=1)
    return outputs

def idwt2d(x, wave='haar'):
    # shape x: (b, h, w, c)
    nc = int(x.shape.dims[3])
    # 小波波形
    w = pywt.Wavelet(wave)
    # 水平低频 垂直低频
    ll = np.outer(w.dec_lo, w.dec_lo)
    # 水平低频 垂直高频
    lh = np.outer(w.dec_hi, w.dec_lo)
    # 水平高频 垂直低频 
    hl = np.outer(w.dec_lo, w.dec_hi)
    # 水平高频 垂直高频
    hh = np.outer(w.dec_hi, w.dec_hi)
    # 卷积核
    core = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
    core[:, :, 0, 0] = ll[::-1, ::-1]
    core[:, :, 0, 1] = lh[::-1, ::-1]
    core[:, :, 0, 2] = hl[::-1, ::-1]
    core[:, :, 0, 3] = hh[::-1, ::-1]
    core = core.astype(np.float32)
    kernel = np.array([core], dtype=np.float32)
    kernel = tf.convert_to_tensor(kernel)
    s = 2 * (len(w.dec_lo) // 2 - 1)
    # 反变换
    with tf.compat.v1.variable_scope('idwt2d'):
        hcA = tf.math.floordiv(tf.shape(x)[1], 2)
        wcA = tf.math.floordiv(tf.shape(x)[2], 2)
        y = []
        for c in range(nc):
            channel = x[:, :, :, c]
            channel = tf.expand_dims(channel, -1)
            cA = channel[:, 0:hcA, 0:wcA, :]
            cH = channel[:, 0:hcA, wcA:, :]
            cV = channel[:, hcA:, 0:wcA, :]
            cD = channel[:, hcA:, wcA:, :]
            temp = tf.concat([cA, cH, cV, cD], axis=-1)
            y.append(temp)
            pass
        # nc * 4
        y = tf.concat(y, axis=-1)
        y3d = tf.expand_dims(y, 1)
        y3d = tf.split(y3d, nc, 4)
        y3d = tf.concat([a for a in y3d], 1)
        output_shape = [tf.shape(y)[0], tf.shape(y3d)[1], \
                        2*(tf.shape(y)[1]-1)+np.shape(ll)[0], \
                        2*(tf.shape(y)[2]-1)+np.shape(ll)[1], 1]
        x3d = tf.nn.conv3d_transpose(y3d, kernel, output_shape=output_shape, padding='VALID', strides=[1, 1, 2, 2, 1])
        outputs = tf.split(x3d, nc, 1)
        outputs = tf.concat([x for x in outputs], 4)
        outputs = tf.reshape(outputs, (tf.shape(outputs)[0], tf.shape(outputs)[2], tf.shape(outputs)[3], nc))
        outputs = outputs[:, s:2*(tf.shape(y)[1]-1)+np.shape(ll)[0]-s, \
                          s:2*(tf.shape(y)[2]-1)+np.shape(ll)[1]-s, :]
        pass
    return outputs

def dwt2dc(x, wave='haar'):
    # b, t, h, w
    nt = x.shape[1]
    x_2c = tf.transpose(x, perm=[0,2,3,1])
    x_2c = tf.concat([tf.math.real(x_2c), tf.math.imag(x_2c)], axis=-1)
    dwtx_2c = dwt2d(x_2c)
    dwtx = tf.complex(dwtx_2c[:,:,:,0:nt], dwtx_2c[:,:,:,nt:2*nt])
    dwtx = tf.transpose(dwtx, perm=[0,3,1,2])
    return dwtx

def idwt2dc(x, wave='haar'):
    # b, t, h, w
    nt = x.shape[1]
    x_2c = tf.transpose(x, perm=[0,2,3,1])
    x_2c = tf.concat([tf.math.real(x_2c), tf.math.imag(x_2c)], axis=-1)
    idwtx_2c = idwt2d(x_2c)
    idwtx = tf.complex(idwtx_2c[:,:,:,0:nt], idwtx_2c[:,:,:,nt:2*nt])
    idwtx = tf.transpose(idwtx, perm=[0,3,1,2])
    return idwtx


"""
tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, [None, None, None, 3], name='inputs')
image = plt.imread('test.jpg')
plt.imshow(image)
plt.show()
x = np.array([image, image[:, ::-1, :]])
dec = wavedec2d(inputs, level=5, wave='sym4')
dwt = dwt2d(inputs, wave='sym4')
idwt = idwt2d(dwt, wave='sym4')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(dec, feed_dict={inputs:x})
    plt.imshow(np.array(result[0], dtype=np.uint8))
    plt.show()
    trans = sess.run(dwt, feed_dict={inputs:x})
    plt.imshow(np.array(trans[0], dtype=np.uint8))
    plt.show()
    recons = sess.run(idwt, feed_dict={inputs:x})
    plt.imshow(np.array(recons[0], dtype=np.uint8))
    plt.show()
    pass
"""