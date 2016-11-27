# chainer-cnn
Code for Convolutional Neural Network for sentence classification (positive or negative) .  

# Requirements
This code is written in Python with Chainer which is framework of Deep Neural Network.  
Please download `GoogleNews-vectors-negative300.bin.gz` from [this site](https://code.google.com/archive/p/word2vec/) and put it in the same directory as these codes.  

# Usage
```
  $ python train_sc-cnn.py [-h] [--gpu CORE_NUMBER] [--data PATH] [--epoch EPOCH]  
  [--batchsize BATCHSIZE] [--save-model PATH] [--save-optimizer PATH] [--baseline]
```

# Optional arguments
```
  -h, --help            show this help message and exit   
  --gpu CORE_NUMBER     use CORE_NUMBER gpu (default: use cpu)  
  --data PATH           an input data file  
  --epoch EPOCH         number of epochs to learn
  --batchsize BATCHSIZE
                        learning minibatch size
  --save-model PATH     save model to PATH
  --save-optimizer PATH
                        save optimizer to PATH
  --baseline            if true, run baseline model
```
