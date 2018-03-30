import tensorflow as tf
import model
## data
image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = \
        datasets.get_dataset(FLAGS.dataset_name,
                             FLAGS.dataset_split_name,
                             FLAGS.dataset_dir,
                             FLAGS.im_batch,
                             is_training=True)

#inputs = tf.placeholder(tf.float32, shape=[1, None, None, 3])

# TODO: preprocess

scores,bbox = model.rpn(image)

# TODO: score loss
#loss_score = scores
#loss_bbox = bbox  # (x,y,h,w)

print(scores)