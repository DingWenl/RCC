# The code will be upload soon

# RCC
Data augmentation for cross-subject SSVEP classification

# The core code for RCC can be found in the `data_generator_source_RCC.py` file within the pretraining directory.

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

## Subject-independent test in the pretraining directory 
1. Run the `pretraining.py` file to get the pre-trained model;
2. Run the `test_independent.py` file to conduct subject-independent test.
   
## Subject-adaptive test in the finetuning directory
1. Run the `finetuning.py` file to finetune the pre-trained model;
2. Run the `test_adaptive.py` file to conduct subject-adaptive test.
