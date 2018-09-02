#!/usr/bin/env python3
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from vgg import vgg_16


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


# 每个像素点有 0 ~ 255 的选择，RGB 三个通道
cm2lbl = np.zeros(256**3)

for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):

    # data = im.astype('int32').asnumpy()这么写会报错:
    # AttributeError: 'numpy.ndarray' object has no attribute 'asnumpy'
    data = im.astype('int32')
    # 用opencv读入的图片按照BRG
    idx = (data[:,:,2]*256+data[:,:,1])*256+data[:,:,0]

    return np.array(cm2lbl[idx])

# 将object_detection.utils.dataset_util下面几个定义TFRecord数据集的工具函数拿过来:
def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def dict_to_tf_example(data, label):
    #data
    ##'maskData/VOCdevkit/VOC2012//JPEGImages/2011_003255.jpg'

    #label
    #maskData/VOCdevkit/VOC2012//SegmentationClass/2011_003255.png

    with open(data, 'rb') as inf:
        #读取图片
        encoded_data = inf.read()
    #读取mask
    img_label = cv2.imread(label)
    
    #定义颜色
    img_mask = image2label(img_label)
    
    #改变编码
    encoded_label = img_mask.astype(np.uint8).tobytes()
   
    #获取高度和宽度
    height, width = img_label.shape[0], img_label.shape[1]

    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        # 小于要求大小的图片全部过滤掉,即 H & W 要大于等于 224
        return None

    # Your code here, fill the dict
    # 提取文件名
    image_name=os.path.splitext(os.path.basename(data))[0]
    #2007_003668

    # 文件名要转化为b'xxx',否则会报错
    # TypeError: 'xxx' has type str, but expected one of: bytes
    image_name=image_name.encode()

    # 在sample-code文件夹中所有文件搜索image/
    # 发现dataset.py中出现了调用:
    # image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    # label = tf.decode_raw(features['image/label'], tf.uint8)
    # 这样就了解了字典应该怎样补充了.

    feature_dict = {
        'image/height': int64_feature(height), # 标签图片高,训练无用的feature
        'image/width': int64_feature(width), # 标签图片宽,训练无用的feature
        'image/filename': bytes_feature(image_name), # 图片名,注意要encode,训练无用的feature
        'image/encoded':  bytes_feature(encoded_data), # 训练图片
        'image/label': bytes_feature(encoded_label), # 标签图片
        'image/format':bytes_feature('jpeg'.encode('utf8')), # 训练无用的feature
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def create_tf_record(output_filename, file_pars):

    """Creates a TFRecord file from examples.
    Args:
        output_filename: tfrecord的保存路径+文件名.
        file_pars: 原图像和标签图像轮机的元组列表.
        列表元素: ('.../JPEGImages/xxx.jpg' '.../SegmentationClass/xxx.png')
    """
    #output_filename
    #maskData/fcn_train.record  和   maskData/fcn_val.record

    #file_pars
    #('maskData/VOCdevkit/VOC2012//JPEGImages/2011_003255.jpg', 'maskData/VOCdevkit/VOC2012//SegmentationClass/2011_003255.png')
  
    #获取到输出流
    writer = tf.python_io.TFRecordWriter(output_filename)

    for img_path, label_path in file_pars:
        # print(img_path,label_path)
        tf_example = dict_to_tf_example(img_path, label_path)
        #img_path
        #'maskData/VOCdevkit/VOC2012//JPEGImages/2011_003255.jpg'
     
        #label_path
        #maskData/VOCdevkit/VOC2012//SegmentationClass/2011_003255.png

        # 只有非none的返回才进行输出,否则
        # AttributeError: 'NoneType' object has no attribute 'SerializeToString'
        if not(tf_example is None):
            #输出
            writer.write(tf_example.SerializeToString())
    #关闭流
    writer.close()

def read_images_names(root, train=True):
    #root maskData/VOCdevkit/VOC2012/
    
    #读取训练集的所有文件的名称的txt文件夹
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')
    
    #打开文件
    with open(txt_fname, 'r') as f:
        #获取文件名
        images = f.read().split()
    #定义图片的文件名列表
    data = []
    #获取到图片mask的位置
    label = []
    for fname in images:
        #添加到列表中
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))

    #返回元祖(data(图片),label(mask))
    ##('maskData/VOCdevkit/VOC2012//JPEGImages/2011_003255.jpg', 'maskData/VOCdevkit/VOC2012//SegmentationClass/2011_003255.png')
    return zip(data, label)


def main(_):
    logging.info('Prepare dataset file names')
    #获取输出的路径
    train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
    #maskData/fcn_train.record
    
    #获取输出的路径
    val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.record')
    #maskData/fcn_val.record
    

    train_files = read_images_names(FLAGS.data_dir, True)
    #FLAGS.data_dir  maskData/VOCdevkit/VOC2012/
    ##返回元祖(data(图片),label(mask))
    ##('maskData/VOCdevkit/VOC2012//JPEGImages/2011_003255.jpg', 'maskData/VOCdevkit/VOC2012//SegmentationClass/2011_003255.png')

    val_files = read_images_names(FLAGS.data_dir, False)
    #FLAGS.data_dir  maskData/VOCdevkit/VOC2012/
    ##返回元祖(data(图片),label(mask))
    ##('maskData/VOCdevkit/VOC2012//JPEGImages/2011_003255.jpg', 'maskData/VOCdevkit/VOC2012//SegmentationClass/2011_003255.png')
  
    #创建训练和测试集
    create_tf_record(train_output_path, train_files)
    create_tf_record(val_output_path, val_files)
   
    #train_output_path
    #maskData/fcn_train.record

    #val_output_path
    #maskData/fcn_val.record

    #train_files
    ###('maskData/VOCdevkit/VOC2012//JPEGImages/2011_003255.jpg', 'maskData/VOCdevkit/VOC2012//SegmentationClass/2011_003255.png')


if __name__ == '__main__':
    tf.app.run()

#python convert_fcn_dataset.py --data_dir=./VOC2012 --output_dir=./
