from keras.callbacks import ModelCheckpoint
from data_generator import train_datagenerator1
import scipy.io as scio 
from scipy import signal
from keras.models import Model,load_model
from keras.layers import Input
import numpy as np
from random import sample
import os
import tensorflow as tf
# from tensorflow.keras.losses import CategoricalCrossentropy
# get the filtered EEG-data, label and the start time of each trial of the dataset
def get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,path):
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
    # open the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    #%% Setting hyper-parameters
    # ampling frequency after downsampling
    fs = 250
    # the number of the electrode channels
    channel = 9
    # the hyper-parameters of the training process
    batchsize = 256
    # the filter ranges of the four sub-filters in the filter bank
    f_down1 = 6
    f_up1 = 50
    wn11 = 2*f_down1/fs
    wn21 = 2*f_up1/fs
    
    f_down2 = 14
    f_up2 = 50
    wn12 = 2*f_down2/fs
    wn22 = 2*f_up2/fs
    
    f_down3 = 22
    f_up3 = 50
    wn13 = 2*f_down3/fs
    wn23 = 2*f_up3/fs

    #%% Training the models of multi-subjects and multi-time-window
    # the list of the time-window
    t_train_list = [1.0]# [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
    # the block index list of different numbers of training blocks, these block index is randomly selected
    total_list = [[[1], [1, 4], [4, 5, 3], [3, 1, 2, 5], [3, 2, 4, 1, 5]], [[0], [3, 4], [5, 3, 0], [0, 5, 4, 2], [5, 0, 4, 3, 2]], [[5], [4, 1], [4, 5, 1], [1, 5, 3, 4], [5, 4, 1, 3, 0]], [[0], [4, 2], [1, 0, 5], [4, 2, 5, 1], [1, 2, 4, 5, 0]], [[3], [2, 1], [5, 0, 2], [1, 0, 5, 3], [2, 0, 1, 5, 3]], [[2], [4, 3], [4, 3, 0], [2, 0, 4, 1], [1, 0, 3, 4, 2]]]
    # selecting the training subject
    for group_n in range(5):
        test_subject_list = list(range(group_n*7+1,(group_n+1)*7+1))
        for sub_selelct in test_subject_list:#sub_list:
        # the path of the dataset and you need change it for your training
            path = '/data/dwl/ssvep/benchmark/S%d.mat'%sub_selelct
            # get the filtered EEG-data of three sub-input, label and the start time of all trials of the training data
            data1, data2, data3 = get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,path)
            # selecting the training time-window
            for t_train in t_train_list:
                # transfer time to frame
                win_train = int(fs*t_train)
                # leave one-block out training
                for test_block in range(6):
                    # different numbers of training blocks 
                    for block_num in range(1,6):
                        train_list = total_list[test_block][block_num-1]
                        val_list = [test_block]
                        train_gen = train_datagenerator1(batchsize,data1, data2, data3,win_train,train_list, channel)
                        #%% the path of pre-trained model
                        pretrianing_model_path = '/data/dwl/ssvep/model/benchmark_RCC_test/cnnformer_RCC/pre_%3.1fs_02_%d.h5'%(t_train,group_n)
                        # load the pre-trained model
                        model = load_model(pretrianing_model_path)
                        # set the number of training epochs
                        train_epoch =  10*block_num
                        # the save path of the fine-tuned model and you need to change it
                        model_path = '/data/dwl/ssvep/model/benchmark_RCC_test/finetune/t_%3.1fs_%dsubject_%dblcok_num_%dtest_block.h5'%(t_train, sub_selelct,block_num,test_block)
                        # some hyper-parameters in the training process
                        model_checkpoint = ModelCheckpoint(model_path, monitor='loss',verbose=1, save_best_only=True,mode='auto')
                        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                        # training
                        history = model.fit_generator(
                                train_gen,
                                steps_per_epoch= 10,
                                epochs=train_epoch,
                                validation_data=None,
                                validation_steps=1,
                                callbacks=[model_checkpoint]
                                )
    # # show the process of the taining
    # epochs=range(len(history.history['loss']))
    # plt.subplot(221)
    # plt.plot(epochs,history.history['accuracy'],'b',label='Training acc')
    # plt.plot(epochs,history.history['val_accuracy'],'r',label='Validation acc')
    # plt.title('Traing and Validation accuracy')
    # plt.legend()
    # # plt.savefig('D:/dwl/code_ssvep/DL/cross_session/m_coyy/photo/model_V3.1_acc1.jpg')
    
    # plt.subplot(222)
    # plt.plot(epochs,history.history['loss'],'b',label='Training loss')
    # plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
    # plt.title('Traing and Validation loss')
    # plt.legend()
    # # plt.savefig('D:/dwl/code_ssvep/DL/cross_session/m_coyy/photo/model_2.5s_loss0%d.jpg'%sub_selelct)
    
    # plt.show()






