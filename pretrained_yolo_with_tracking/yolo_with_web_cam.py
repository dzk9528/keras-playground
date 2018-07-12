# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:15:10 2018

The test program with video capture for pretrained yolo, the output path is in sample2_output
@author: zhang
"""
import collections
import cv2
import numpy as np

from keras import backend as K
from PIL import Image
from track_sys import yolo_model_config, draw_bbx_class, detect_and_track


def main():
    # Load model and tensor from model path
    score_threshold, iou_threshold = 0.3, 0.5
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes, yolo_model, model_image_size, class_names=yolo_model_config(input_image_shape,
                                                                                       score_threshold, 
                                                                                       iou_threshold,
                                                                                       debug = False)
    print("Initialzation completed.")
    
    tracker_list=[[], [], []]
    memory = collections.defaultdict(int)
    
    #initialize parameters
    parameter = {}
    parameter["boxes"] = boxes
    parameter["scores"] = scores
    parameter["classes"] = classes
    parameter["class_names"] = class_names
    parameter["dis_threshold"] = 200
    parameter["model_image_size"] = model_image_size
    parameter["input_image_shape"] = input_image_shape
    parameter["is_numpy"] = True
    
    sess = K.get_session()
    print("Session started")
    track_start = False
    
    cap = cv2.VideoCapture(0)
        
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if track_start:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            out_boxes, out_classes = detect_and_track(sess, 
                                                  yolo_model,
                                                  frame, 
                                                  parameter,
                                                  tracker_list,
                                                  memory)
            #print(out_boxes.shape, out_classes.shape)
            frame = draw_bbx_class(frame, out_boxes, out_classes, class_names)
            frame = np.array(frame)[:,:,[2,1,0]]
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('g'):
            track_start = True
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
        
    sess.close()
    
if __name__=="__main__":
    main()