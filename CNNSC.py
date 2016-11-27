#encoding: utf8

from chainer import ChainList
import chainer.functions as F
import chainer.links as L

"""
Code for Convolutional Neural Networks for Sentence Classification

author: ichiroex
"""

# リンク数を可変にしたいのでChainListを使用する
class CNNSC(ChainList):
    def __init__(self,
                 input_channel,
                 output_channel,
                 filter_height,
                 filter_width,
                 n_label,
                 max_sentence_len):
        # フィルター数、使用されたフィルター高さ、最大文長は後から使う
        self.cnv_num = len(filter_height)
        self.filter_height = filter_height
        self.max_sentence_len = max_sentence_len
        
        # Convolution層用のLinkをフィルター毎に追加
        # Convolution2D(　入力チャンネル数, 出力チャンネル数（形毎のフィルターの数）, フィルターの形（タプル形式で）, パディングサイズ )
        link_list = [L.Convolution2D(input_channel, output_channel, (i, filter_width), pad=0) for i in filter_height]
        # Dropoff用のLinkを追加
        link_list += [L.Linear(output_channel * self.cnv_num, output_channel * self.cnv_num)]
        # 出力層へのLinkを追加
        link_list += [L.Linear(output_channel * self.cnv_num, n_label)]

        # ここまで定義したLinkのリストを用いてクラスを初期化する
        super(CNNSC, self).__init__(*link_list)
        
        # ちなみに
        # self.add_link(link)
        # みたいにリンクを列挙して順々に追加していってもOKです

    def __call__(self, x, train=True):
        # フィルタを通した中間層を準備
        h_conv = [None for _ in self.filter_height]
        h_pool = [None for _ in self.filter_height]
        
        # フィルタ形毎にループを回す
        for i, filter_size in enumerate(self.filter_height):
            # Convolition層を通す
            h_conv[i] = F.relu(self[i](x))
            # Pooling層を通す
            h_pool[i] = F.max_pooling_2d(h_conv[i], (self.max_sentence_len+1-filter_size))
        # Convolution+Poolingを行った結果を結合する
        concat = F.concat(h_pool, axis=2)
        # 結合した結果に対してDropoutをかける
        h_l1 = F.dropout(F.tanh(self[self.cnv_num+0](concat)), ratio=0.5, train=train)
        # Dropoutの結果を出力層まで圧縮する
        y = self[self.cnv_num+1](h_l1)

        return y

if __name__ == '__main__':
    model = L.Classifier(CNNSC(input_channel=1,
                           output_channel=100,
                           filter_height=[3,4,5],
                           filter_width=20,
                           n_label=2,
                           max_sentence_len=20))
    print('done process')
