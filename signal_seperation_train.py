from keras.models import Sequential, load_model
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.optimizers import Adam
import collections
import numpy as np
import os
import random


signal1 = [1.0 for i in range(100)]
signal2 = [1.0 for i in range(100)]
source = [[(i+j)] for i,j in zip(signal1, signal2)]
target = [[i,j] for i,j in zip(signal1, signal2)]



def get_next_batch(num_frequencies, batch_size):
    X = None
    Y = None
    return X, Y

if __name__ == "__main__":   
############################# main ###############################

    # initialize parameters
    DATA_DIR = ""
    BATCH_SIZE = 32
    WINDOW_LENGTH = 10
    FREQUENCY_BINS = 100 # number of frequencies
    KERNEL = (3,2)
    FILTERS = 128
    KERNEL_OUTPUT = (3,1)
    OUTPUT_CHANNELS = 2
    
    EPOCHS_TRAIN = 1000
    
    modelFile = "signal-seperation-network.h6"

    if os.path.exists(modelFile):
        model = load_model(os.path.join(DATA_DIR, modelFile))
    else:
        # build the model
        model = Sequential()
        # read the phase and frequency and also read the neighbors values
        # 128 different evaluations of neighboring filters
        # padding='valid' and return_sequences=False
        # input is (Batch, Time, 100, 2, 1) the output will be (Batch, 98, 1, 128)
        model.add(
            ConvLSTM2D(
                filters=FILTERS,
                kernel_size=KERNEL,
                strides=(1,1),
                padding='valid',
                data_format='channels_last',
                activation='relu',
                batch_input_shape=(BATCH_SIZE,WINDOW_LENGTH,FREQUENCY_BINS,2,1),
                return_sequences=True))
        model.add(
            ConvLSTM2D(
                filters=OUTPUT_CHANNELS,
                kernel_size=KERNEL_OUTPUT,
                strides=(1,1),
                padding='valid',
                data_format='channels_last',
                activation='relu',
                batch_input_shape=(BATCH_SIZE,WINDOW_LENGTH,FREQUENCY_BINS-4,1,FILTERS),
                return_sequences=False))
    model.summary()

    model.compile(optimizer=Adam(lr=1e-6), loss="mse")

    # train network
    fout = open(os.path.join(DATA_DIR, "blackjack-simple-results.tsv"), "w")
    
    for e in range(EPOCHS_TRAIN):
        loss = 0.0

        X, Y = get_next_batch(FREQUENCY_BINS, BATCH_SIZE)
        loss += model.train_on_batch(X, Y)

        if e % 20 == 0:
            print("\nEpoch {:04d}/{:d} | Loss {:.5f}\n" 
                  .format(e+1, EPOCHS_TRAIN, loss), end="\n")

        fout.write("{:04d}\t{:.5f}\n" 
                   .format(e+1, loss))

        if e % 100 == 0:
            model.save(os.path.join(DATA_DIR, modelFile), overwrite=True)
                
    print("Done")
    fout.close()
    model.save(os.path.join(DATA_DIR, modelFile), overwrite=True)
