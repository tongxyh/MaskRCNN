#gt_boxes: an array of shape(Gx5), [x1, y1, x2, y2,class ]
#all_anchors: an array of shape  (h, w, A, 4),
import numpy as np

def anchor_generator(x,y):
    anchors = []
    for scale in [128,256,512]:
        for ratio in [0.5,1,2]:
            anchor = [x,y,scale,scale*ratio]
            anchors.append(anchor)
    return anchors

def anchor_plane(width,height,stride):
    K = 9
    all_anchors = []

    for x in range(width):
        for y in range(height):
            #shift
            anchors = anchor_generator(0, 0)
            for i in range(K):
                anchors[i][0] = anchors[i][0]  + x*stride
                anchors[i][1] = anchors[i][1] + y*stride
            all_anchors.append(anchors)
    return all_anchors

def overlap(anchor,gt_bbox):
    x_relat = gt_bbox[0] - anchor[0]
    y_relat = gt_bbox[1] - anchor[1]
    count = 0
    for w in range(gt_bbox[2]):
        for h in range(gt_bbox[3]):
            if w+x_relat>=0 and w+x_relat<anchor[2] and h+y_relat>=0 and h+y_relat<anchor[3]:
                count = count + 1
    return count*1.0/anchor[2]/anchor[3]

def anchor_sample(anchors,gt_bboxes):
    #boundry limitation

    for anchor in anchors:
        #calculate overlap
        ious = []
        for gt_bbox in gt_bboxes:
            iou = overlap(anchor,gt_bbox[:4])
            ious.append(iou)

            if(iou > 0.7):
                pass
                #positive
    pass

#test
#anchor_plane(40,60,16)