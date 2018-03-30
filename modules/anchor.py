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
    bg = np.zeros((bg_width,bg_height),np.uint8)
    x_gt,y_gt,dx_gt,dy_gt = gt_bbox
    x_an, y_an, dx_an, dy_an = anchor
    bg[int(x_gt -0.5 * dx_gt): int(x_gt + 0.5 * dx_gt), int(y_gt - 0.5 * dy_gt):int(y_gt + 0.5*dy_gt) ] = bg[int(x_gt -0.5 * dx_gt): int(x_gt + 0.5 * dx_gt), int(y_gt - 0.5 * dy_gt):int(y_gt + 0.5*dy_gt) ] + 1
    bg[int(x_an - 0.5 * dx_an):int(x_an + 0.5 * dx_an), int(y_an - 0.5 * dy_an):int(y_an + 0.5 * dy_an)] = bg[int(x_an - 0.5 * dx_an):int(x_an + 0.5 * dx_an), int(y_an - 0.5 * dy_an):int(y_an + 0.5 * dy_an)] + 1

    over_pixels = np.sum(bg[bg == 2.]) * 0.5
    ALL = np.sum(bg) - over_pixels
    IoU = over_pixels/ ALL

    #plt.imshow(bg)
    #plt.title(IoU)
    #plt.show()
    #print(IoU)
    #overlap check finished

    return IoU

class samples():
    def __init__(self,anchor = None,gt_bbox = None,iou = 0,front = False):
        self.anchor = anchor
        self.gt_bbox = gt_bbox
        self.iou = iou
        self.front = front

def anchor_sample(all_anchors,gt_bboxes):

    # positive - (1) highest IoU (2)Iou > 0.7
    # negative - (1) Iou < 0.3
    # gt_bbox - [A,5]
    # all_anchors [N,K,4]

    count_pos = 0
    count_neg = 0
    ious = [] # all_samples
    ious_neg = [] # negative samples
    ious_max = [] #s amples with largest IoU for each gt_bbox
    for gt_bbox in gt_bboxes:
        #time_beg = time.time()
        max_iou_sample = samples()
        count=0
        for anchors in all_anchors:
            for anchor in anchors:
                iou_max = 0.0
                count += 1
                #if(count%10000 == 0):
                #    print(count)
                # boundry limitation
                # if anchor[0] + anchor[2] < width and anchor[1] + anchor[3] < height
                if(count_pos < 128 or count_neg < 128):
                    iou = overlap(anchor,gt_bbox[:4])
                else:
                    ious.extend(ious_neg[:(256-len(ious))])
                    print("Postive Samples:", count_pos, "Negative Samples:", 256 - count_pos, "All Samples:", len(ious))
                    #print(time.time() - time_beg, "s")
                    return ious
                #print(iou)
                if iou > iou_max:
                    iou_max = iou
                    max_iou_sample.anchor = anchor
                    max_iou_sample.gt_bbox = gt_bbox
                    max_iou_sample.iou = iou
                    max_iou_sample.True = True

                if(iou > 0.7):
                    # anchor_idx,anchor,gt_bbox,iou,1
                    ious.append(samples(anchor,gt_bbox,iou,True))
                    count_pos += 1
                if(iou < 0.3 and count_neg < 256 - count_pos):
                    ious_neg.append(samples(anchor,gt_bbox,iou,False))
                    count_neg += 1
        if(max_iou_sample.iou > 0.3):
            ious_max.append(max_iou_sample)
        #print(time.time() - time_beg, "s")
    if(count_pos < 128):
        ious.extend(ious_max[:max(128-count_pos,len(ious_max))])
    ious.extend(ious_neg[:(256-len(ious))])
    print("Postive Samples:",count_pos,"Negative Samples:",256 - count_pos,"All Samples:", len(ious))
    return ious

#test
