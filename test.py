import tensorflow as tf
import os
from model import LplusS_Net, S_Net, SLR_Net
from dataset_tfrecord import get_dataset
import argparse
import scipy.io as scio
import mat73
import numpy as np
from datetime import datetime
import time
from tools.tools import video_summary, mse, tempfft



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', metavar='str', nargs=1, default=['test'], help='training or test')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'], help='batch size')
    parser.add_argument('--niter', metavar='int', nargs=1, default=['10'], help='number of network iterations')
    parser.add_argument('--acc', metavar='int', nargs=1, default=['16'], help='accelerate rate')
    parser.add_argument('--net', metavar='str', nargs=1, default=['SLRNet'], help='SLR Net or S Net')
    parser.add_argument('--weight', metavar='str', nargs=1, default=['models/stable/2020-10-28T10-14-41SLRNET_DYNAMIC_V2_MULTICOIL16/epoch-50/ckpt'], help='modeldir in ./models')
    parser.add_argument('--gpu', metavar='int', nargs=1, default=['5'], help='GPU No.')
    parser.add_argument('--data', metavar='str', nargs=1, default=['DYNAMIC_V2_MULTICOIL'], help='dataset name')
    parser.add_argument('--learnedSVT', metavar='bool', nargs=1, default=['True'], help='Learned SVT threshold or not')

    args = parser.parse_args()

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)

    dataset_name = args.data[0].upper()
    mode = args.mode[0]
    batch_size = int(args.batch_size[0])
    niter = int(args.niter[0])
    acc = int(args.acc[0])
    net_name = args.net[0].upper()
    weight_file = args.weight[0]
    learnedSVT = bool(args.learnedSVT[0])

    print('network: ', net_name)
    print('acc: ', acc)
    print('load weight file from: ', weight_file)


    result_dir = os.path.join('results/stable', weight_file.split('/')[2] + net_name + str(acc) + '_lr_0.001')
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    logdir = './logs'
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    summary_writer = tf.summary.create_file_writer(os.path.join(logdir, mode, TIMESTAMP + net_name + str(acc) + '/'))

    # prepare undersampling mask
    if dataset_name == 'DYNAMIC_V2':
        multi_coil = False
        mask_size = '18_192_192'
    elif dataset_name == 'DYNAMIC_V2_MULTICOIL':
        multi_coil = True
        mask_size = '18_192_192'
    elif dataset_name == 'FLOW':
        multi_coil = False
        mask_size = '20_180_180'

    """
    if acc == 8:
        mask = scio.loadmat('mask_newdata/cartesian_' + mask_size + '_acs4_acc8.mat')['mask']
    elif acc == 10:
        mask = scio.loadmat('mask_newdata/cartesian_' + mask_size + '_acs4_acc10.mat')['mask']
    elif acc == 12:
        mask = scio.loadmat('mask_newdata/cartesian_' + mask_size + '_acs4_acc12.mat')['mask']
    """
    if acc == 8:
        mask = mat73.loadmat('/data1/ziwenke/SLRNet/mask_newdata/vista_' + mask_size + '_acc_8.mat')['mask']
    elif acc == 10:
        mask = mat73.loadmat('/data1/ziwenke/SLRNet/mask_newdata/vista_' + mask_size + '_acc_10.mat')['mask']
    elif acc == 12:
        mask = mat73.loadmat('/data1/ziwenke/SLRNet/mask_newdata/vista_' + mask_size + '_acc_12.mat')['mask']

    
    mask = tf.cast(tf.constant(mask), tf.complex64)

    # prepare dataset
    dataset = get_dataset(mode, dataset_name, batch_size, shuffle=False)
    
    # initialize network
    if net_name == 'SLRNET':
        net = SLR_Net(mask, niter, learnedSVT)

        

    net.load_weights(weight_file)
    
    # Iterate over epochs.
    for i, sample in enumerate(dataset):
        # forward
        
        k0 = None
        csm = None
        #with tf.GradientTape() as tape:
        if multi_coil:
            k0, label, csm = sample
        else:
            k0, label = sample
        label_abs = tf.abs(label)

        k0 = k0 * mask

        
        t0 = time.time()
        recon, X_SYM = net(k0, csm)
        t1 = time.time()
    
        recon_abs = tf.abs(recon)

        loss_total = mse(recon, label)

        tf.print(i, 'mse =', loss_total.numpy(), 'time = ', t1-t0)

        result_file = os.path.join(result_dir, 'recon_'+str(i+1)+'.mat')
        
        datadict = {'recon': np.squeeze(tf.transpose(recon, [0,2,3,1]).numpy())}
        scio.savemat(result_file, datadict)

        # record gif
        with summary_writer.as_default():
            combine_video = tf.concat([label_abs[0:1,:,:,:], recon_abs[0:1,:,:,:]], axis=0).numpy()
            combine_video = np.expand_dims(combine_video, -1)
            video_summary('convin-'+str(i+1), combine_video, step=1, fps=10)


