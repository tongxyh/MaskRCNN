#gt_boxes: an array of shape(Gx5), [x1, y1, x2, y2,class ]
#all_anchors: an array of shape  (h, w, A, 4),
import numpy as np
import time
from matplotlib import pyplot as plt

def anchor_generator(x,y):
    anchors = []
    for scale in [128,256,512]:
        for ratio in [0.5,1,2]:
            anchor = [x,y,scale,scale*ratio]
            anchor[0] = x + 0.5*scale
            anchor[1] = y + 0.5*scale*ratio
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

    bg_width = int(max(gt_bbox[0] + 0.5*gt_bbox[2] , anchor[0] + 0.5*anchor[2]))
    bg_height = int(max(gt_bbox[1] + 0.5*gt_bbox[3] , anchor[1] + 0.5*anchor[3]))
    bg = np.zeros((bg_width,bg_height))
    x_gt,y_gt,dx_gt,dy_gt = gt_bbox
    x_an, y_an, dx_an, dy_an = anchor
    bg[int(x_gt -0.5 * dx_gt): int(x_gt + 0.5 * dx_gt), int(y_gt - 0.5 * dy_gt):int(y_gt + 0.5*dy_gt) ] = bg[int(x_gt -0.5 * dx_gt): int(x_gt + 0.5 * dx_gt), int(y_gt - 0.5 * dy_gt):int(y_gt + 0.5*dy_gt) ] + 1.
    bg[int(x_an - 0.5 * dx_an):int(x_an + 0.5 * dx_an), int(y_an - 0.5 * dy_an):int(y_an + 0.5 * dy_an)] = bg[int(x_an - 0.5 * dx_an):int(x_an + 0.5 * dx_an), int(y_an - 0.5 * dy_an):int(y_an + 0.5 * dy_an)] + 1.

    ALL = np.sum(bg) - np.sum(bg[bg == 2.]) * 0.5
    IoU = np.sum(bg[bg == 2.])*0.5/ ALL
    
    #plt.imshow(bg)
    #plt.title(IoU)
    #plt.show()
    #print(IoU)
    #overlap check finished

    return IoU

    '''
    count = 0
    x_relat = gt_bbox[0] - anchor[0]
    y_relat = gt_bbox[1] - anchor[1]
    for w in range(gt_bbox[2]):
        for h in range(gt_bbox[3]):
            if w+x_relat>=0 and w+x_relat<anchor[2] and h+y_relat>=0 and h+y_relat<anchor[3]:
                count = count + 1U
    return count*1.0/anchor[2]/anchor[3]
    '''

def anchor_sample(all_anchors,gt_bboxes):


    # positive - (1) highest IoU (2)Iou > 0.7
    # negative - (1) Iou < 0.3
    # gt_bbox - [A,5]
    # all_anchors [N,K,4]

    time_beg = time.time()

    count_pos = 0
    count_neg = 0

    for gt_bbox in gt_bboxes:
        #calculate overlap
        #print(anchor)
        ious = []
        for anchors in all_anchors:
            for anchor in anchors:

                # boundry limitation
                # if anchor[0] + anchor[2] < width and anchor[1] + anchor[3] < height
                iou = overlap(anchor,gt_bbox[:4])
                #print(iou)
                if(iou > 0.7):
                    #print(iou)
                    ious.append([anchor,gt_bbox,iou,1])
                    count_pos += 1
                    #print(count_pos)

                if(iou < 0.3):
                    ious.append([anchor, gt_bbox, iou, 0])
                    count_neg += 1

    print("ALL POSITIVE ANCHORS NUM:",count_neg,count_pos)
    print(time.time() - time_beg,"s")
#test
