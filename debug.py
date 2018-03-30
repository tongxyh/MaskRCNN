#import os
#os.system("export CUDA_VISIBLE_DEVICES=1")

import tensorflow as tf
from modules import anchor
from utili.datasets import dataset_factory as datasets
import utili.config as config
from matplotlib import pyplot as plt

from modules import anchor

FLAGS = tf.app.flags.FLAGS
all_anchors = anchor.anchor_plane(40,60,16)

image, ih, iw, gt_boxes, gt_masks, num_instances, img_id, iterator = \
        datasets.get_dataset(FLAGS.dataset_name,
                             FLAGS.dataset_split_name,
                             FLAGS.dataset_dir,
                             FLAGS.im_batch,
                             is_training=False)

stride = 8


with tf.Session() as sess:
    sess.run(iterator.initializer)
    for i in range(100):
        gt_bboxes,height,width = sess.run([gt_boxes,ih,iw])

all_anchors = anchor.anchor_plane(width/stride,height/stride,stride = 8)
anchor_s = anchor.anchor_sample(all_anchors,gt_boxes)

# TODO: bbox regression loss
#anchor_s

# TODO: score loss

#plt.imshow(abs(image[0]))
#plt.show()