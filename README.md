# RCC
Data augmentation for cross-subject SSVEP classification

## The core code of EEG-ME in data_generator.py
# reshape the mixing ration list
index_style = np.reshape(index_list,(batchsize,1,1))
# obtain original training samples
x_original = np.array(x_train)
# create the list to save the decorrelated sample and eigenvector matrix
featVec_list,x_decorrelated_list = list(range(batchsize)), list(range(batchsize))
for i in range(batchsize):
    # reshape i-th training samples to facilitate the application of RCC
    x_data = np.reshape(x_original[i],(channel, win_train*3))
    # obtain the i-th covariance matrix
    x_cov = np.cov(x_data)
    # obtain the i-th eigenvector matrix
    _, featVec=np.linalg.eig(x_cov)
    # obtain the decorrelated sample
    x_decorrelated = np.dot(featVec.T,x_data) #np.linalg.inv(featVec1),featVec1.T
    # save the decorrelated sample and eigenvector matrix
    x_decorrelated_list[i] = x_decorrelated
    featVec_list[i] = featVec
# obtain the random selected U_j
featVec_random = np.copy(featVec_list)
random.shuffle(featVec_random)

featVec_list = np.array(featVec_list)
featVec_random = np.array(featVec_random)
# obtain the mixed eigenvector matrix
featVec_list = index_style*featVec_list + (1-index_style)*featVec_random
# create list to save the reconstructed samples
recon_list = list(range(batchsize))
for i in range(batchsize):
    # obtain the i-th reconstructed sample
    recon_x = np.dot(featVec_list[i],x_decorrelated_list[i])
    # reshape the i-th reconstructed sample to be the input of the deep learnning model
    recon_x = np.reshape(recon_x,(channel, win_train,3))
    # save the i-th reconstructed sample
    recon_list[i] = recon_x
# obtain the mini-batch training samples and labels
x_input = np.array(recon_list)



## The related version information
1. Python == 3.7.0
2. Keras-gpu == 2.3.1
3. tensorflow-gpu == 2.1.0
4. scipy == 1.5.2
5. numpy == 1.19.2
## Training CNN-Former model with RCC for the benchmark dataset
1. Download the code.
2. Download the [benchmark dataset](http://bci.med.tsinghua.edu.cn/download.html) and its [paper](https://ieeexplore.ieee.org/abstract/document/7740878).
3. Create a model folder to save the model.
4. Change the data and model folder paths in train and test files to your data and model folder paths.

