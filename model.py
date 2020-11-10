import tensorflow as tf
from tensorflow.keras import layers
import os
import numpy as np
import time
from tools.tools import tempfft, fft2c_mri, ifft2c_mri, Emat_xyt


class CNNLayer(tf.keras.layers.Layer):
    def __init__(self, n_f=32):
        super(CNNLayer, self).__init__()
        self.mylayers = []

        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.LeakyReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        self.mylayers.append(tf.keras.layers.LeakyReLU())
        self.mylayers.append(tf.keras.layers.Conv3D(2, 3, strides=1, padding='same', use_bias=False))
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        if len(input.shape) == 4:
            input2c = tf.stack([tf.math.real(input), tf.math.imag(input)], axis=-1)
        else:
            input2c = tf.concat([tf.math.real(input), tf.math.imag(input)], axis=-1)
        res = self.seq(input2c)
        res = tf.complex(res[:,:,:,:,0], res[:,:,:,:,1])
        
        return res

class CONV_OP(tf.keras.layers.Layer):
    def __init__(self, n_f=32, ifactivate=False):
        super(CONV_OP, self).__init__()
        self.mylayers = []
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        if ifactivate == True:
            self.mylayers.append(tf.keras.layers.ReLU())
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        res = self.seq(input)
        return res

class SLR_Net(tf.keras.Model):
    def __init__(self, mask, niter, learned_topk=False):
        super(SLR_Net, self).__init__(name='SLR_Net')
        self.niter = niter
        self.E = Emat_xyt(mask)
        self.learned_topk = learned_topk
        self.celllist = []
    

    def build(self, input_shape):
        for i in range(self.niter-1):
            self.celllist.append(SLRCell(input_shape, self.E, learned_topk=self.learned_topk))
        self.celllist.append(SLRCell(input_shape, self.E, learned_topk=self.learned_topk, is_last=True))

    def call(self, d, csm):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        """
        if csm == None:
            nb, nt, nx, ny = d.shape
        else:
            nb, nc, nt, nx, ny = d.shape
        X_SYM = []
        x_rec = self.E.mtimes(d, inv=True, csm=csm)
        t = tf.zeros_like(x_rec)
        beta = tf.zeros_like(x_rec)
        x_sym = tf.zeros_like(x_rec)
        data = [x_rec, x_sym, beta, t, d, csm]
        
        for i in range(self.niter):
            data = self.celllist[i](data, d.shape)
            x_sym = data[1]
            X_SYM.append(x_sym)

        x_rec = data[0]
        
        return x_rec, X_SYM


class SLRCell(layers.Layer):
    def __init__(self, input_shape, E, learned_topk=False, is_last=False):
        super(SLRCell, self).__init__()
        if len(input_shape) == 4:
            self.nb, self.nt, self.nx, self.ny = input_shape
        else:
            self.nb, nc, self.nt, self.nx, self.ny = input_shape

        self.E = E
        self.learned_topk = learned_topk
        if self.learned_topk:
            if is_last:
                self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=False, name='thres_coef')
                self.eta = tf.Variable(tf.constant(0.01, dtype=tf.float32), trainable=False, name='eta')
            else:
                self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True, name='thres_coef')
                self.eta = tf.Variable(tf.constant(0.01, dtype=tf.float32), trainable=True, name='eta')

        self.conv_1 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_2 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_3 = CONV_OP(n_f=16, ifactivate=False)
        self.conv_4 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_5 = CONV_OP(n_f=16, ifactivate=True)
        self.conv_6 = CONV_OP(n_f=2, ifactivate=False)
        #self.conv_7 = CONV_OP(n_f=16, ifactivate=True)
        #self.conv_8 = CONV_OP(n_f=16, ifactivate=True)
        #self.conv_9 = CONV_OP(n_f=16, ifactivate=True)
        #self.conv_10 = CONV_OP(n_f=16, ifactivate=True)

        self.lambda_step = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='lambda_1')
        self.lambda_step_2 = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='lambda_2')
        self.soft_thr = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='soft_thr')


    def call(self, data, input_shape):
        if len(input_shape) == 4:
            self.nb, self.nt, self.nx, self.ny = input_shape
        else:
            self.nb, nc, self.nt, self.nx, self.ny = input_shape
        x_rec, x_sym, beta, t, d, csm = data

        t = self.lowrank(x_rec)
        x_rec, x_sym = self.sparse(x_rec, d, t, beta, csm)
        
        beta = self.beta_mid(beta, x_rec, t)

        data[0] = x_rec
        data[1] = x_sym
        data[2] = beta
        data[3] = t

        return data

    def sparse(self, x_rec, d, t, beta, csm):
        lambda_step = tf.cast(tf.nn.relu(self.lambda_step), tf.complex64)
        lambda_step_2 = tf.cast(tf.nn.relu(self.lambda_step_2), tf.complex64)

        ATAX_cplx = self.E.mtimes(self.E.mtimes(x_rec, inv=False, csm=csm) - d, inv=True, csm=csm)

        r_n = x_rec - tf.math.scalar_mul(lambda_step, ATAX_cplx) +\
              tf.math.scalar_mul(lambda_step_2, x_rec + beta - t)

        # D_T(soft(D_r_n))
        if len(r_n.shape) == 4:
            r_n = tf.stack([tf.math.real(r_n), tf.math.imag(r_n)], axis=-1)

        x_1 = self.conv_1(r_n)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)

        x_soft = tf.math.multiply(tf.math.sign(x_3), tf.nn.relu(tf.abs(x_3) - self.soft_thr))

        x_4 = self.conv_4(x_soft)
        x_5 = self.conv_5(x_4)
        x_6 = self.conv_6(x_5)

        x_rec = x_6 + r_n

        x_1_sym = self.conv_4(x_3)
        x_1_sym = self.conv_5(x_1_sym)
        x_1_sym = self.conv_6(x_1_sym)
        #x_sym_1 = self.conv_10(x_1_sym)

        x_sym = x_1_sym - r_n
        x_rec = tf.complex(x_rec[:, :, :, :, 0], x_rec[:, :, :, :, 1])

        return x_rec, x_sym

    def lowrank(self, x_rec):
        [batch, Nt, Nx, Ny] = x_rec.get_shape()
        M = tf.reshape(x_rec, [batch, Nt, Nx*Ny])
        St, Ut, Vt = tf.linalg.svd(M)
        if self.learned_topk:
            #tf.print(tf.sigmoid(self.thres_coef))
            thres = tf.sigmoid(self.thres_coef) * St[:, 0]
            thres = tf.expand_dims(thres, -1)
            St = tf.nn.relu(St - thres)
        else:
            top1_mask = np.concatenate(
                [np.ones([self.nb, 1], dtype=np.float32), np.zeros([self.nb, self.nt - 1], dtype=np.float32)], 1)
            top1_mask = tf.constant(top1_mask)
            St = St * top1_mask
        St = tf.linalg.diag(St)
        
        St = tf.dtypes.cast(St, tf.complex64)
        Vt_conj = tf.transpose(Vt, perm=[0, 2, 1])
        Vt_conj = tf.math.conj(Vt_conj)
        US = tf.linalg.matmul(Ut, St)
        M = tf.linalg.matmul(US, Vt_conj)
        x_rec = tf.reshape(M, [batch, Nt, Nx, Ny])

        return x_rec

    def beta_mid(self, beta, x_rec, t):
        eta = tf.cast(tf.nn.relu(self.eta), tf.complex64)
        return beta + tf.multiply(eta, x_rec - t)

class S_Net(tf.keras.Model):
    def __init__(self, mask, niter):
        super(S_Net, self).__init__(name='S_Net')
        self.niter = niter
        self.E = Emat_xyt(mask)

        self.celllist = []
    
    def build(self, input_shape):
        for i in range(self.niter-1):
            self.celllist.append(SCell_learned_step(input_shape, self.E, is_last=False))
        self.celllist.append(SCell_learned_step(input_shape, self.E, is_last=True))

    def call(self, d):
        nb, nt, nx, ny = d.shape
        Spre = tf.reshape(self.E.mtimes(d, inv=True), [nb, nt, nx*ny])
        Mpre = Spre

        data = [Spre, Mpre, d]

        for i in range(self.niter):
            data = self.celllist[i](data)
        
        S, M, _ = data
        #M = tf.reshape(M, [nb, nt, nx, ny])
        S = tf.reshape(S, [nb, nt, nx, ny])

        return S


class SCell_learned_step(layers.Layer):

    def __init__(self, input_shape, E, is_last):
        super(SCell_learned_step, self).__init__()
        self.nb, self.nt, self.nx, self.ny = input_shape
        
        self.E = E

        self.sconv = CNNLayer(n_f=32)

        self.is_last = is_last
        if not is_last:
            self.gamma = tf.Variable(tf.constant(1, dtype=tf.float32), trainable=True)

    def call(self, data):
        Spre, Mpre, d = data

        S = self.sparse(Mpre)
        
        dc = self.dataconsis(S, d)
        if not self.is_last:
            gamma = tf.cast(tf.nn.relu(self.gamma), tf.complex64)
        else:
            gamma = tf.cast(1.0, tf.complex64)
        M = S - gamma * dc
        
        data[0] = S
        data[1] = M

        return data

    def sparse(self, S):
        S = tf.reshape(S, [self.nb, self.nt, self.nx, self.ny])
        S = self.sconv(S)
        S = tf.reshape(S, [self.nb, self.nt, self.nx*self.ny])

        return S

    def dataconsis(self, LS, d):
        resk = self.E.mtimes(tf.reshape(LS, [self.nb, self.nt, self.nx, self.ny]), inv=False) - d
        dc = tf.reshape(self.E.mtimes(resk, inv=True), [self.nb, self.nt, self.nx*self.ny])
        return dc
