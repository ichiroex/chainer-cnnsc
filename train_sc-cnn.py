# -*- coding: utf-8 -*-

import sys, os
import six
import argparse
import numpy as np
from sklearn.cross_validation import train_test_split

import chainer
import chainer.links as L
from chainer import optimizers, cuda, serializers
import chainer.functions as F

from CNNSC import CNNSC
import util

"""
Code for the paper, Convolutional Neural Networks for Sentence Classification (EMNLP2014)

CNNによるテキスト分類 (posi-nega)
"""

def get_parser():
    
    DEF_GPU = -1
    DEF_DATA = "..{sep}Data{sep}input.dat".format(sep=os.sep)
    DEF_EPOCH = 100
    DEF_BATCHSIZE = 50

    #引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        dest='gpu',
                        type=int,
                        default=DEF_GPU,
                        metavar='CORE_NUMBER',
                        help='use CORE_NUMBER gpu (default: use cpu)')
    parser.add_argument('--data',
                        dest='data',
                        type=str,
                        default=DEF_DATA,
                        metavar='PATH',
                        help='an input data file')
    parser.add_argument('--epoch',
                        dest='epoch',
                        type=int,
                        default=DEF_EPOCH,
                        help='number of epochs to learn')
    parser.add_argument('--batchsize',
                        dest='batchsize',
                        type=int,
                        default=DEF_BATCHSIZE,
                        help='learning minibatch size')
    parser.add_argument('--save-model',
                        dest='save_model',
                        action='store',
                        type=str,
                        default=None,
                        metavar='PATH',
                        help='save model to PATH')
    parser.add_argument('--save-optimizer',
                        dest='save_optimizer',
                        action='store',
                        type=str,
                        default=None,
                        metavar='PATH',
                        help='save optimizer to PATH')
    parser.add_argument('--baseline',
                        dest='baseline',
                        action='store_true',
                        help='if true, run baseline model')

    return parser

def save_model(model, file_path='sc_cnn.model'):
    # modelを保存
    print 'save the model'
    model.to_cpu()
    serializers.save_npz(file_path, model)

def save_optimizer(optimizer, file_path='sc_cnn.state'):
    # optimizerを保存
    print 'save the optimizer'
    serializers.save_npz(file_path, optimizer)

def train(args):

    batchsize   = args.batchsize    # minibatch size
    n_epoch     = args.epoch        # エポック数

    # Prepare dataset
    dataset, height, width = util.load_data(args.data)
    #dataset, height, width = util.load_data_with_rand_vec(args.data)
    
    print 'height (max length of sentences):', height
    print 'width (size of wordembedding vecteor ):', width

    dataset['source'] = dataset['source'].astype(np.float32) #特徴量
    dataset['target'] = dataset['target'].astype(np.int32) #ラベル

    x_train, x_test, y_train, y_test = train_test_split(dataset['source'], dataset['target'], test_size=0.10)
    N_test = y_test.size         # test data size
    N = len(x_train)             # train data size
    in_units = x_train.shape[1]  # 入力層のユニット数 (語彙数)

    # (nsample, channel, height, width) の4次元テンソルに変換
    input_channel = 1
    x_train = x_train.reshape(len(x_train), input_channel, height, width) 
    x_test  = x_test.reshape(len(x_test), input_channel, height, width)

    n_label = 2 # ラベル数
    filter_height = [3,4,5] # フィルタの高さ
    baseline_filter_height = [3]
    filter_width  = width # フィルタの幅 (embeddingの次元数)
    output_channel = 100 
    decay = 0.0001 # 重み減衰
    grad_clip = 3  # gradient norm threshold to clip
    max_sentence_len = height # max length of sentences

    # モデルの定義
    if args.baseline == False:
        # 提案モデル
        model = CNNSC(input_channel,
                      output_channel,
                      filter_height,
                      filter_width,
                      n_label,
                      max_sentence_len)
    else:
        # ベースラインモデル (フィルタの種類が１つ)
        model = CNNSC(input_channel,
                      output_channel,
                      baseline_filter_height,
                      filter_width,
                      n_label,
                      max_sentence_len)
 
    # Setup optimizer
    optimizer = optimizers.AdaDelta()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

    #GPUを使うかどうか
    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy #args.gpu <= 0: use cpu, otherwise: use gpu

    # Learning loop
    for epoch in six.moves.range(1, n_epoch + 1):

        print 'epoch', epoch, '/', n_epoch
        
        # training
        perm = np.random.permutation(N) #ランダムな整数列リストを取得
        sum_train_loss     = 0.0
        sum_train_accuracy = 0.0
        for i in six.moves.range(0, N, batchsize):

            #perm を使い x_train, y_trainからデータセットを選択 (毎回対象となるデータは異なる)
            x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]])) #source
            t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]])) #target
            
            model.zerograds()

            y = model(x)
            loss = F.softmax_cross_entropy(y, t) # 損失の計算
            accuracy = F.accuracy(y, t) # 正解率の計算
            
            sum_train_loss += loss.data * len(t)
            sum_train_accuracy += accuracy.data * len(t)
            
            # 最適化を実行
            loss.backward()
            optimizer.update()

        print('train mean loss={}, accuracy={}'.format(sum_train_loss / N, sum_train_accuracy / N)) #平均誤差

        # evaluation
        sum_test_loss     = 0.0
        sum_test_accuracy = 0.0
        for i in six.moves.range(0, N_test, batchsize):

            # all test data
            x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]))
            t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]))
            
            y = model(x, False)
            loss = F.softmax_cross_entropy(y, t) # 損失の計算
            accuracy = F.accuracy(y, t) # 正解率の計算
 
            sum_test_loss += loss.data * len(t)
            sum_test_accuracy += accuracy.data * len(t)

        print(' test mean loss={}, accuracy={}'.format(sum_test_loss / N_test, sum_test_accuracy / N_test)) #平均誤差

        sys.stdout.flush()
        
    return model, optimizer

def main():
    parser = get_parser()
    args = parser.parse_args()
    model, optimizer = train(args)

    if args.save_model != None:
        save_model(model)
    if args.save_optimizer != None:
        save_optimizer(optimizer)

if __name__ == "__main__":
    main()

