from random import sample
import random
import numpy as np
from keras.utils import np_utils
# get the training sampels
def train_datagenerator1(batchsize,train_data1,train_data2,train_data3,win_train,train_list, channel):
    while True:
        x_train, y_train = list(range(batchsize)), list(range(batchsize))
        target_list = list(range(40))
        for i in range(int(batchsize)):
            k = sample(train_list, 1)[0]
            m = sample(target_list, 1)[0]
            
            # randomly selecting a single-sample in the single-trial, 35 is the frames of the delay-time
            time_start = random.randint(35+125,int(1250+35+125-win_train))
            time_end = time_start + win_train
            # get three sub-inputs
            x_11 = train_data1[k][m][:,time_start:time_end]
            x_21 = np.reshape(x_11,(channel, win_train, 1))
            
            
            x_12 = train_data2[k][m][:,time_start:time_end]
            x_22 = np.reshape(x_12,(channel, win_train, 1))
            

            x_13 = train_data3[k][m][:,time_start:time_end]
            x_23 = np.reshape(x_13,(channel, win_train, 1))

            x_concatenate = np.concatenate((x_21, x_22, x_23), axis=-1)
            x_train[i] = x_concatenate
            y_train[i] = np_utils.to_categorical(m, num_classes=40, dtype='float32')
                
        x_input = np.array(x_train)
        y_input = np.reshape(y_train,(batchsize,40))

        yield x_input, y_input




