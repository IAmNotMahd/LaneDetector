# Jorge Reyes, Nikita Turley, Mahd Amir Khan

#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os.path as osp
import os
import random



# for tensorboard
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
LOGDIR = "{}/run-{}/".format(root_logdir,now)


from tensorflow.contrib import layers
from skimage.io import imread, imsave

DATASET_PATH = '/home/jorge/Documents/cos429/cos429challenge'
BATCH_SIZE = 16
EPOCH_NUMBER = 15
DUMP_FOLDER = '/home/jorge/Documents/cos429/cos429challenge/prediction'
NUM_IM = 20000

if not osp.exists(DUMP_FOLDER):
    os.makedirs(DUMP_FOLDER)


############ RESIDUAL MODULE #########################################
def residual_module(data, numFilters, stride, chanDim, isTrain,reduce=False,
    reg=0.0001, bnEps=3e-5, bnMom = 0.9):
    
    
    # initialise skip connection data
    skip = data

    # Using ResNetXt BottleNeck BoResidual Architecture
    # -> 1x1X64 Conv -> 3X3X64 Conv -> 1X1X256Conv
    # https://arxiv.org/pdf/1512.03385.pdf

    # Architecture used
    # https://arxiv.org/pdf/1606.00373.pdf

    # Resnet module module implementation based on one used Adrian's Rosenbrock
    # is his Deep Learning Book and Aurelien Geron's Hand's on
    # Machine learning book


    # first convolution block of residual (1X1 Conv)
    bn1 = tf.layers.batch_normalization(data, axis = chanDim, momentum = bnMom,
        epsilon = bnEps, training = isTrain)
    act1 = tf.nn.relu(bn1)
    conv1 = tf.layers.conv2d(inputs=act1,filters=int(numFilters/4),kernel_size=[1,1],strides=1,
        padding="same", use_bias=False, kernel_regularizer = tf.contrib.layers.l2_regularizer(reg))
    
    # second block of residual (3x3 conv)
    bn2 = tf.layers.batch_normalization(conv1, axis = chanDim, momentum = bnMom,
        epsilon = bnEps, training = isTrain)
    act2 = tf.nn.relu(bn2)
    conv2 = tf.layers.conv2d(inputs=act2,filters=int(numFilters/4),kernel_size=[3,3],strides=stride,
        padding="same", use_bias=False, kernel_regularizer = tf.contrib.layers.l2_regularizer(reg))

    # third block of residual (1x1 conv)
    bn3 = tf.layers.batch_normalization(conv2, axis = chanDim, momentum = bnMom,
        epsilon = bnEps, training=isTrain)
    act3 = tf.nn.relu(bn3)
    conv3 = tf.layers.conv2d(inputs=act3,filters=numFilters,kernel_size=[1,1],strides=1,
        padding="same", use_bias=False, kernel_regularizer = tf.contrib.layers.l2_regularizer(reg))


    # if we are reducing spatial size, also need to apply a convolution
    # to the skip path to match dimmensions
    if (reduce == True):
        skip = tf.layers.conv2d(inputs=skip,filters=numFilters,kernel_size=[1,1],strides=stride,
        padding="same", use_bias=False, kernel_regularizer = tf.contrib.layers.l2_regularizer(reg))


    # Element wise addition of channels
    output = tf.add(conv3, skip)


    return output



def build_model(isTrain=False):


    color = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
    mask = tf.placeholder(dtype=tf.bool, shape=[None, 128, 128])
    target = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])

    stages = [3, 4, 6, 3]
    filters = [256, 512, 1024, 2048]
    reg = 0.0001
    bnEps=3e-5 
    bnMom=0.9

    
    # apply mask to input image
    input = color

    print("Input Size")
    print(input.get_shape())

    # since channel dimmension (not the same as batch) is last dimmenion,
    # "channels last"
    chanDim = -1

    # Layer 1: 64, 7X7, Stride of 2 Convolution
    x = tf.layers.conv2d(inputs=input,filters=64,kernel_size=[7,7],strides=2,
        padding="same", kernel_regularizer = tf.contrib.layers.l2_regularizer(reg))
    print("Size After Conv 1")
    print(x.get_shape())


    # Layer 2: Max Pool 64, 3x3, + 2(S) 
    x = tf.layers.max_pooling2d(inputs=x,pool_size=[3,3],strides=2)
    
    print("Size Before Residuals")
    print(x.get_shape())

    # loop over stages (residual blocks)
    for i in range(0, len(stages)):
        #initialize stride, then apply residual module
        # used to reduce the spatial size of the input volume
        if i == 0:
            stride = 1
        else:
            stride = 2

        print("Size Before Stage %d" % i)
        print(x.get_shape())
        # apply a reduction at the beginning of every stage 
        x = residual_module(x, filters[i], stride,
            chanDim, isTrain, reduce=True, reg=0.0001, bnEps=2e-5, bnMom = 0.9)
        print("Size After Stage %d" % i)
        print(x.get_shape())

        # loop over the number of layers in every stage
        # Note: -1 is due to account to already created residual 
        # unit that reduced the size in the line above
        for j in range(0, stages[i]-1):
            # add a Residual Unit
            x = residual_module(x, filters[i],1, chanDim, isTrain,
                bnEps=bnEps,bnMom=bnMom)
            print("Size afer Stage %d:%d"%(i,j))
            print(x.get_shape())

    x = tf.layers.conv2d(inputs=x,filters=2048,kernel_size=[1,1],strides=1,
        padding="same", kernel_regularizer = tf.contrib.layers.l2_regularizer(reg))
    
    x = tf.layers.batch_normalization(x, axis = chanDim, momentum = bnMom,
        epsilon = bnEps, training=isTrain)

    print("BottleNeck Size")
    print(x.get_shape())

    # loop over upstages (up pooling)
    pairs = (len(filters)-1)
    for i in range(len(stages)+1, 0,-1):
        if (pairs < 0):
            pairs = 0
        x = tf.layers.conv2d_transpose(x, filters[pairs], [5,5], 2, 'same',  activation=tf.nn.relu)
        print("Up Pool Size")
        print(x.get_shape())
        pairs = pairs - 1
       


    x = tf.layers.conv2d(inputs=x,filters=3,kernel_size=[3,3],strides=1,
        padding="same", kernel_regularizer = tf.contrib.layers.l2_regularizer(reg))
    predict = x

    print("Output Size")
    print(predict.get_shape())


    predict_n = tf.nn.l2_normalize(predict, axis=3)
    target_n = tf.nn.l2_normalize(target, axis=3)

    cosine_angle = tf.reduce_sum(predict_n * target_n, axis=3)

    loss = -tf.reduce_mean(tf.boolean_mask(cosine_angle, mask))


    return color, mask, target, predict_n, loss


#def load_train_data(iteration, batch_size):
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
    target_npy = np.zeros([total, 128, 128, 3], dtype=np.float32)

    
    #for i in range(batch_size):
    for i in range(total):
        color_path = osp.join(DATASET_PATH, 'train', 'color', '{}.png'.format(batchIndex[i]))
        mask_path = osp.join(DATASET_PATH, 'train', 'mask', '{}.png'.format(batchIndex[i]))
        target_path = osp.join(DATASET_PATH, 'train', 'normal', '{}.png'.format(batchIndex[i]))
        color_npy[i, ...] = imread(color_path)
        mask_npy[i, ...] = imread(mask_path, as_gray=True)
        target_npy[i, ...] = imread(target_path)

    

    target_npy = target_npy / 255.0 * 2 - 1

    return color_npy, mask_npy, target_npy


def train():
    color, mask, target, predict_n, loss = build_model(isTrain=True)

    # to write loss summaries
    loss_summ = tf.summary.scalar('training_loss', loss)
    writer = tf.summary.FileWriter(LOGDIR, tf.get_default_graph()) 

    # to save model at checkpoints (IMPORTANT: each ckpt file is about 1GB)
    saver = tf.train.Saver(max_to_keep=15)

    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
           
            # Max Iteration is 
            count = 0
            for i in range(EPOCH_NUMBER):
                print("Epoch " + str(i))

                # Shuffle batch indicies at every epoch
                s = np.arange(NUM_IM)
                np.random.shuffle(s)
            
                for index in range(0,NUM_IM-16,16):
                    count = count + 1

                    batchIndex = s[index:index+BATCH_SIZE]
                    color_npy,mask_npy, target_npy = load_train_data(batchIndex)

                    feed_dict = {
                    color: color_npy,
                    mask: mask_npy,
                    target: target_npy
                    }
                    
                    loss_val, summ, _ = sess.run([loss, loss_summ, train_op], feed_dict=feed_dict)
                    writer.add_summary(summ, count)

                    print(str(i) + " " + str(loss_val))
                
                # save every epoch 
                save_path = saver.save(sess, "tmp/resnet_{}.ckpt".format(i))

            # save last model 
            save_path = saver.save(sess,"tmp/resnet_final.ckpt")

            # close FileWriter (writes loss summaries)
            writer.close()

            print(" Finished Training")

if __name__ == '__main__':
    train()
