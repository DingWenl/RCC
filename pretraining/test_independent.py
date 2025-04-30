# import keras
from keras.models import load_model
import scipy.io as scio 
import random
import numpy as np
from scipy import signal
import os
from random import sample
from keras.utils import np_utils
import pandas as pd
import tensorflow as tf
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# get the training sampels
def datagenerator(batchsize,train_data1,train_data2,train_data3,win_train,channel,test_list):
    x_train, y_train = list(range(batchsize)), list(range(batchsize))
    target_list = list(range(40))
    # get training samples of batchsize trials
    for i in range(batchsize):
        k = test_list[0]
        m = sample(target_list, 1)[0]
        # randomly selecting a single-sample in the single-trial, 35 is the frames of the delay-time
        time_start = random.randint(int(35+125),int(1250+35+125-win_train))
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
    
    return x_input, y_input

# get the filtered EEG-data, label and the start time of each trial of the dataset (test set), more details refer to the "get_train_data" in "FB-tCNN_train"
def get_test_data(wn11,wn21,wn12,wn22,wn13,wn23,path):
    # read the data
    data = scio.loadmat(path)
    # get the EEG-data of the selected electrodes and downsampling it
    data_1 = data['data']
    c1 = [47,53,54,55,56,57,60,61,62]
    
    train_data = data_1[c1,:,:,:]
    # get the filtered EEG-data with six-order Butterworth filter of the first sub-filter
    block_data_list1 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn11,wn21], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list1.append(target_data_list)
    # get the filtered EEG-data with six-order Butterworth filter of the second sub-filter
    block_data_list2 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn12,wn22], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list2.append(target_data_list)

    block_data_list3 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn13,wn23], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list3.append(target_data_list) 
    return block_data_list1, block_data_list2, block_data_list3

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # Setting hyper-parameters, more details refer to "pretraining"
    fs = 250
    channel = 9
    # number of test smaples for the target subject
    batchsize = 5000
    f_down1 = 6
    f_up1 = 50#18
    wn11 = 2*f_down1/fs
    wn21 = 2*f_up1/fs
    
    f_down2 = 14
    f_up2 = 50#34
    wn12 = 2*f_down2/fs
    wn22 = 2*f_up2/fs
    
    f_down3 = 22
    f_up3 = 50
    wn13 = 2*f_down3/fs
    wn23 = 2*f_up3/fs
    
    
    total_av_acc_list = []
    t_train_list = [1.0] # [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]

    for group_n in range(5):
        test_subject_list = list(range(group_n*7+1,(group_n+1)*7+1))
        for sub_selelct in test_subject_list:#sub_list:
            # the path of the dataset and you need to change it for your test
            path = '/data/dwl/ssvep/benchmark/S%d.mat'%sub_selelct
            # get the filtered EEG-data of the three sub-filters, label and the start time of all trials of the test data
            data1, data2, data3 = get_test_data(wn11,wn21,wn12,wn22,wn13,wn23,path)
            av_acc_list = []
            for t_train in t_train_list:
                win_train = int(fs*t_train)
                # the path of the trained model and you need to change it for your test
                model_path = '/data/dwl/ssvep/model/benchmark_RCC_test/cnnformer_RCC/pre_%3.1fs_02_%d.h5'%(t_train,group_n)
                model = load_model(model_path)
                print("load successed")
                print(t_train, sub_selelct)
                acc_list = []
                # all the blocks from the target subject utilized as the test set
                for block_test in range(6):
                    test_list = [block_test]
                    # get the filtered EEG-data and label of the test samples
                    x_train,y_train = datagenerator(batchsize,data1, data2, data3, win_train, channel,test_list)
                    a, b = 0, 0
                    y_pred = model.predict(np.array(x_train))
                    # Calculating the accuracy of current block
                    for i in range (batchsize):
                        y_pred_ = np.argmax(y_pred[i])
                        y_true_  = np.argmax(y_train[i])
                        if y_true_ == y_pred_:
                            a += 1
                        else:
                            b+= 1
                    acc = a/(a+b)
                    acc_list.append(acc)
                    print('block ',block_test, acc)
                av_acc = np.mean(acc_list)
                print('av:',av_acc)
                av_acc_list.append(av_acc)
            total_av_acc_list.append(av_acc_list)

        
        
        
        # save the results
        # print(total_av_acc_list)
    company_name_list = total_av_acc_list
    df = pd.DataFrame(company_name_list)
    df.to_excel("/home/Dwl/result/test_%3.1fs.xlsx"%t_train, index=False)
