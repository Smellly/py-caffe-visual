#coding=utf-8 
       
import os
import caffe 
import numpy as np 

from time import clock
from tqdm import tqdm

# 根目录 
root = '/data/users/jensenjwang/work/ssd/caffe/'
# deploy文件 
deploy = root + 'jobs/liveness/senet/senet_deploy.prototxt'
# 训练好的 caffemodel 
caffe_model = root + \
        'jobs/liveness/senet/snapshots/senet-scale4-binhai2-2nd_iter_12000.caffemodel'

def forward(deploy, caffe_model, filelist):
    # 加载model和network 
    net = caffe.Net(deploy, caffe_model, caffe.TEST)

    # 图片预处理设置 #设定图片的shape格式(1,3,64,64) 
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28) 
    transformer.set_transpose('data', (2,0,1)) 
    # 减去均值，
    transformer.set_mean('data', np.array([104, 117, 124]))
    # 缩放到[0, 255]之间 
    transformer.set_raw_scale('data', 255)
    # 交换通道，将图片由RGB变为BGR 
    transformer.set_channel_swap('data', (2,1,0)) 

    preds = []

    # caffe.set_device(2)
    caffe.set_mode_gpu()

    for i in range(0, len(filelist)):
        img = filelist[i]
        # 加载图片   
        im = caffe.io.load_image(img)
        # 执行上面设置的图片预处理操作，并将图片载入到blob中 
        # print('blobd : ', im)
        net.blobs['data'].data[...] = transformer.preprocess('data', im)

        # 执行测试 
        # out = net.forward() 
        out, diffs = net.forward_backward_all()
           
        # 取出最后一层（prob）属于某个类别的概率值，并打印,'prob'为最后一层的名称
        blobd = net.blobs['data'].data[0].flatten()
        prob0, prob1 = net.blobs['prob'].data[0].flatten()
        pool5 = net.blobs['pool5'].data[0].flatten()
        pool5_diff = out_back#.blobs['pool5'].diff
        
        # 'prob': array([[  5.39986126e-04,   9.99460042e-01]], dtype=float32)
        print('blobd : ', blobd)
        print('pool5 : ', pool5)
        print('pool5_diff : ', pool5_diff['data'])
        print('pool5_diff shape: ', pool5_diff['data'].shape)
        # print len(blobd)
        print('prob : ', prob0, prob1)
        preds.append(' '.join(str(e) for e in blobd) + '\n')
        preds.append('%f %f\n'%(prob0, prob1))
    return preds

def main():
    testfile = '/data/users/jensenjwang/work/ssd/caffe/data/liveness/zby_test_1400_1800_neg.txt'
    with open(testfile, 'r') as f:
        raw = f.read().split('\n')[:-1]

    filelist = [x.split()[0] for x in raw]
    labellist = [x.split()[1] for x in raw]

    # length = len(filelist)
    length = 1
    start = clock()
    preds = forward(deploy, caffe_model, filelist[:length])
    stop = clock()

    print('Used %d s'%(stop - start))

if __name__ == '__main__':
    main()



