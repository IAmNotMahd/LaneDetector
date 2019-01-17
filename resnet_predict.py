#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os.path as osp
import os

from resnet import build_model
from skimage.io import imread, imsave

DATASET_PATH = '/home/jorge/Documents/cos429/cos429challenge'
DUMP_FOLDER = '/home/jorge/Documents/cos429/cos429challenge/test/prediction'
# DUMP_FOLDER = '/home/jorge/Documents/cos429/cos429challenge/prediction'


NUM_IM = 2000

def load_train_data(batchIndex):

    # To handle one by one batches
    if (type(batchIndex) is int):
        tmp = batchIndex
        batchIndex = np.arange(1)
        batchIndex[0] = tmp


    total = batchIndex.shape[0]

    # creating arrays of zeros to store color, mask, and target
    color_npy = np.zeros([total, 128, 128, 3], dtype=np.float32)
    mask_npy = np.zeros([total, 128, 128], dtype=np.uint8)
    # target_npy = np.zeros([total, 128, 128, 3], dtype=np.float32)

    
    #for i in range(batch_size):
    for i in range(total):
        color_path = osp.join(DATASET_PATH, 'test', 'color', '{}.png'.format(batchIndex[i]))
        mask_path = osp.join(DATASET_PATH, 'test', 'mask', '{}.png'.format(batchIndex[i]))
        # target_path = osp.join(DATASET_PATH, 'train', 'normal', '{}.png'.format(batchIndex[i]))
        color_npy[i, ...] = imread(color_path)
        mask_npy[i, ...] = imread(mask_path, as_gray=True)
        # target_npy[i, ...] = imread(target_path)

    

    # target_npy = target_npy / 255.0 * 2 - 1

    return color_npy, mask_npy #, target_npy

def predict():
    # delete any existing graphs
    tf.reset_default_graph()

    # create the graph by loading model
    color, mask, target, predict_n, loss = build_model()

    # create the saver object
    saver = tf.train.Saver()

    with tf.device('/gpu:0'):
            with tf.Session() as sess:

                # initialize all of the variables
                sess.run(tf.global_variables_initializer())

                # restore model
                saver.restore(sess,"tmp/resnet_10.ckpt")
                print("Model Restored")

                # Output Predictions
                print("Running Predictions")
                for i in range(NUM_IM):
                    print("Picture: " + str(i))
                    color_npy, mask_npy = load_train_data(i)
                    feed_dict = {
                        color: color_npy,
                        mask: mask_npy,
                        #target: target_npy
                    }
                    predict_val, = sess.run([predict_n], feed_dict={color: color_npy})
                    predict_img = ((predict_val.squeeze(0) + 1) / 2 * 255).astype(np.uint8)
                    imsave(osp.join(DUMP_FOLDER, '{}.png'.format(i)), predict_img)

if __name__ == '__main__':
    predict()