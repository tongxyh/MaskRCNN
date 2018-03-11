import tensorflow as tf

def cls_score_layer(input):
    score = tf.keras.layers.Conv2D(18,(1,1),activation=tf.nn.softmax,name="rpn_cls_score")
    return score

def bbox_pred_layer(input):
    bbox = tf.keras.layers.Conv2D(36,(1,1),name="rpn_bbox_pred")
    return bbox

def anchor()
#import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp
chkp.print_tensors_in_checkpoint_file("data/resnet_v1_50.ckpt",tensor_name='',all_tensors=False) #set False to only print tensor name and shape

with tf.name_scope("rpn"):
    raw_feat = resnet_v1_50(inputs,num_classes=None,is_training=True,global_pool=True,output_stride=None,reuse=None,scope='resnet_v1_50') #13x13x256

    #conv+relu
    features = tf.keras.layers.Conv2D(256, (3, 3), padding='same',activation='relu',name = "conv_afer_resnet")(raw_feat)

    score = cls_score(features) # (N,H,W,Ax2)
    bbox = bbox_pred(features) # (N,H,W,Ax4)

    # TODO: reshpe score

    loss_score = score
    loss_bbox =
#saver = tf.train.Saver()
#saver.restore()
