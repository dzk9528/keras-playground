"""
The test program fro the analyze frame, the output path is in sample2_output
@author: zhang
"""
import os
import argparse
import time
import collections
import cv2

from keras import backend as K
from PIL import Image
from track_sys import yolo_model_config, draw_bbx_class, detect_and_track

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/input/', type=str)
parser.add_argument('--output_path', default='/output/', type=str)

def main():
    
    args = parser.parse_args()
    
    # Load model and tensor from model path
    score_threshold, iou_threshold = 0.3, 0.5
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes, yolo_model, model_image_size, class_names=yolo_model_config(input_image_shape,
                                                                                       score_threshold, 
                                                                                       iou_threshold,
                                                                                       debug = False)
    # Path setting
    data_path = os.getcwd() + args.data_path
    save_path = os.getcwd() + args.output_path
    
    sess = K.get_session()
    print("Initialzation completed.")
    
    start=time.time()
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
    
    for filename in os.listdir(data_path):
        image = Image.open(os.path.join(data_path, filename))
        
        #image = cv2.imread(os.path.join(data_path, filename))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = Image.fromarray(image)
        
        out_boxes, out_classes = detect_and_track(sess, 
                                              yolo_model,
                                              image, 
                                              parameter,
                                              tracker_list,
                                              memory)
        #print(out_boxes.shape, out_classes.shape)
        image = draw_bbx_class(image, out_boxes, out_classes, class_names)
        image.save(save_path + filename, quality = 90)
        print("image ", filename, " process completed.")
        

    # Without saving the images, the screen will print out: Processing completed in 6.3358 seconds for 34 images
    print("Processing completed in %s seconds for %s images" %( round(time.time() - start, 4), len(os.listdir(data_path)) ) )
    sess.close()
    
    
if __name__=="__main__":
    main()