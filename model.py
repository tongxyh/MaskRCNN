import tensorflow as tf

def resnet_v50():
    pass

def raw_feature_extractor():
    #ZF or VGG16
    pass
#import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file("data/resnet_v1_50.ckpt",tensor_name='',all_tensors=False) #set False to only print tensor name and shape

saver = tf.train.Saver()
saver.restore()
