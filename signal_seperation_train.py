from keras.models import Sequential, load_model
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.optimizers import Adam
from keras import losses
import collections
import numpy as np
import os
import random
import tensorflow as tf
import util
import time
import matplotlib.pyplot as plt

signal1 = [1.0 for i in range(100)]
signal2 = [1.0 for i in range(100)]
source = [[(i+j)] for i,j in zip(signal1, signal2)]
target = [[i,j] for i,j in zip(signal1, signal2)]



def get_next_batch(num_frequencies, batch_size):
    X = None
    Y = None
    return X, Y

def minimum_mse(y_true, y_pred):
    loss1 = losses.mean_squared_error(y_true,y_pred)
    loss2 = losses.mean_squared_error(1-y_true,y_pred)
    loss = tf.minimum(loss1,loss2)
    return loss
	
if __name__ == "__main__":   
############################# main ###############################

    # initialize parameters
# initialize parameters
    DATA_DIR = ""
    BATCH_SIZE = 50
    WINDOW_LENGTH = 40
    FREQUENCY_BINS = 128 # number of frequencies
    KERNEL_INPUT = (3,3)
    CHANNELS_INPUT = 32
    KERNEL_HIDDEN0 = (3,2)
    KERNEL_HIDDEN = (3,1)
    CHANNELS_HIDDEN1 = 16
    CHANNELS_HIDDEN2 = 8
    KERNEL_OUTPUT = (3,1)
    CHANNELS_OUTPUT = 1
    EPOCHS_TRAIN = 1
    DROPOUT = .3
    modelFile = "signal-seperation-network.h6"
    if os.path.exists(modelFile):
        model = load_model(os.path.join(DATA_DIR, modelFile),custom_objects={'minimum_mse': minimum_mse})
    else:
        # build the model
        model = Sequential()
        # read the phase and frequency and also read the neighbors values
        # 128 different evaluations of neighboring filters
        # padding='valid' and return_sequences=False
        # input is (Batch, Time, 100, 2, 1) the output will be (Batch, 98, 1, 128)
        model.add(
            ConvLSTM2D(
                filters=CHANNELS_INPUT,
                kernel_size=KERNEL_INPUT,
                strides=(1,1),
                padding='same',
                data_format='channels_last',
                activation='relu',
                batch_input_shape=(BATCH_SIZE,WINDOW_LENGTH,FREQUENCY_BINS,2,1),
                return_sequences=True))
        model.add(
            ConvLSTM2D(
                filters=CHANNELS_HIDDEN1,
                kernel_size=KERNEL_HIDDEN0,
                strides=(1,1),
                dropout=DROPOUT,
                padding='valid',
                data_format='channels_last',
                activation='relu',
                batch_input_shape=(BATCH_SIZE,WINDOW_LENGTH,FREQUENCY_BINS-2,1,CHANNELS_INPUT),
                return_sequences=True))
        model.add(
            ConvLSTM2D(
                filters=CHANNELS_HIDDEN2,
                kernel_size=KERNEL_HIDDEN,
                strides=(1,1),
                dropout=DROPOUT,
                padding='valid',
                data_format='channels_last',
                activation='relu',
                batch_input_shape=(BATCH_SIZE,WINDOW_LENGTH,FREQUENCY_BINS-4,1,CHANNELS_HIDDEN1),
                return_sequences=True))
        model.add(
            ConvLSTM2D(
                filters=CHANNELS_OUTPUT,
                kernel_size=KERNEL_OUTPUT,
                strides=(1,1),
                dropout=DROPOUT,
                padding='valid',
                data_format='channels_last',
                activation='sigmoid',
                batch_input_shape=(BATCH_SIZE,WINDOW_LENGTH,FREQUENCY_BINS-6,1,CHANNELS_HIDDEN2),
                return_sequences=True))
    model.summary()

    model.compile(optimizer=Adam(lr=1e-10), loss=minimum_mse)

    # train network
    fout = open(os.path.join(DATA_DIR, "signal-seperation-results.tsv"), "w")
    
    
    
    for e in range(EPOCHS_TRAIN):
        st = time.clock()

        loss = 0.0

        X, Y = util.load_random_batch(BATCH_SIZE,WINDOW_LENGTH,FREQUENCY_BINS)
        #print(X.shape)
        #print(Y.shape)
        loss += model.train_on_batch(X, Y)

        if e % 5 == 0:
            t = time.clock()-st
            print("Epoch {:04d}/{:d} | Loss {:.5f} | Time {:.5f}" 
                  .format(e+1, EPOCHS_TRAIN, loss, t), end="\n")

        fout.write("{:04d}\t{:.5f}\n" 
                   .format(e+1, loss))

        if e % 100 == 0:
            model.save(os.path.join(DATA_DIR, modelFile), overwrite=True)
                
    gengraphs = True
    
    if(gengraphs):
        X, Y_M = util.load_random_batch(32,WINDOW_LENGTH,FREQUENCY_BINS)
        print(X.shape)
        plt.imshow(np.swapaxes(X[0,:,:,0,0], 0,1))
        plt.show()
        plt.imshow(np.swapaxes(Y_M[0,:,:,0,0], 0,1))
        plt.show()
        
    print("Done")
    fout.close()
    model.save(os.path.join(DATA_DIR, modelFile), overwrite=True)
