import tensorflow as tf
import utili.dataset_factory as datasets
from modules import resnet_v1
import utili.config as config

def cls_score_layer(input):
    score = tf.keras.layers.Conv2D(18,(1,1),activation=tf.nn.softmax,name="rpn_cls_score")(input)
    score = tf.reshape(score, [-1, 2])  # -1 means flatten to 1-D
    score = tf.nn.softmax(score)
    return score

def bbox_pred_layer(input):
    bbox = tf.keras.layers.Conv2D(36,(1,1),name="rpn_bbox_pred")(input)
    return bbox

#import the inspect_checkpoint library
#from tensorflow.python.tools import inspect_checkpoint as chkp
#chkp.print_tensors_in_checkpoint_file("data/resnet_v1_50.ckpt",tensor_name='',all_tensors=False) #set False to only print tensor name and shape

def rpn(inputs):
    with tf.name_scope("rpn"):

        #raw_feat,end_points = resnet_v1.resnet_v1_50(inputs,num_classes=None,is_training=True,global_pool=True,output_stride=None,reuse=None,scope='resnet_v1_50') #13x13x256
    
        #conv+relu
        features = tf.keras.layers.Conv2D(256, (3, 3), padding='same',activation='relu',name = "conv_afer_resnet")(inputs)

        scores = cls_score_layer(features) # (N,H,W,Ax2) A=9
        bbox = bbox_pred_layer(features) # (N,H,W,Ax4) A=9
    return scores,bbox

#saver = tf.train.Saver()
#saver.restore()
