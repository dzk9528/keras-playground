"""
The functions for the frame_anaysis
@author: zhang
"""
import numpy as np
import cv2

from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
'''
Using the pre-trained model and processing functions from github
    https://github.com/allanzelener/YAD2K
'''
from yolo_utils import read_classes, read_anchors
from yad2k.yad2k.models.keras_yolo import yolo_head, yolo_eval

def yolo_model_config(input_image_shape, 
                      score_threshold, 
                      iou_threshold, 
                      debug=False):
    """
    This function is used to load the model and get the relative parameters in the model
    Arguments:
        input_image_shape (tf placeholder shape: (2, ) dtype: int32): 
            The shape of the image in the form of [frame.size[1], frame.size[0]]
        score_threshold (float): 
            The score threshold for the maximun score of the output class of each region
        iou_threshold (float): 
            The threshold of intersction of union in the Non-max suppression
        debug (bool): 
            The debug flag for me to debug and peek the data and model
    Return:
        boxes (tf tensor shape: (2, ) dtype: int32):
            The tensor ot keep the output boxes after the detection of yolo model
        scores (tf tensor shape: (2, ) dtype: int32):
            The tensor ot keep the output scores for each boxes after the detection of yolo model
        classes (tf tensor shape: (2, ) dtype: int32):
            The tensor ot keep the output classes for each boxes after the detection of yolo model
        yolo_model (keras model): 
            The pretrained keras yolo model
        model_image_size (tuple dtype: int):
            The input size for the pretrained yolo model
        class_names (list dtype: string):
            The class_names of coco datasets
    """
    
    #Load pretrained model and parameters from yad2k
    class_names = read_classes("yad2k/model_data/coco_classes.txt")
    anchors = read_anchors("yad2k/model_data/yolov2_anchors.txt")
    yolo_model = load_model("yad2k/model_data/yolov2.h5")
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    
    
    if debug:
        print('The class names is ', class_names)
        print('The acnchors are ', anchors)
        print('model_image_size=', model_image_size)
        yolo_model.summary()
        
    #Get the output boxes, scores, classes tensor to keep the result
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold)
    
    return boxes, scores, classes, yolo_model, model_image_size, class_names

def MiddlePointDistance(box1, box2): 
    '''
    Get the distance between the middle points of two bounding boxes
    Arguments:
        box1 (list dtype: int): 
            the first boundingbox
        box1 (list dtype: int): 
            the second boundingbox
    Return:
        distance between the middle points of two bounding boxes
    '''
    m1 = np.array([(box1[2]+box1[0])/2.0, (box1[3]+box1[1])/2.0])
    m2 = np.array([(box2[2]+box2[0])/2.0, (box2[3]+box2[1])/2.0])
    return np.sqrt(np.sum((m1-m2)*(m1-m2)))
    
def draw_bbx_class(image, 
                   out_boxes, 
                   out_classes, 
                   class_names, 
                   colors = None, 
                   debug = False):
    """
    Modified version for drawing part the test_yolo.py, which draw bounding boxes, the class names 
    as well as the score on the PIL image
    Arguments:
        image (PIL image shape: (h, w, channel) dtype: uint8): 
            The image where the bounding box the class names as well as the scores hould be drawed on.
        out_boxes (tf tensor shape: (2, ) dtype: int32):
            The output boxes after the detection of yolo model
        out_classes (tf tensor shape: (2, ) dtype: int32):
            The output classes for each boxes after the detection of yolo model
        class_names (list dtype: string):
            The class_names of coco datasets
        color(tuple dtype； unit8):
            the color yu pick for the bounding box as well as the text. e.g: colors=(0, 0, 255)
        debug (bool): 
            The debug flag for me to debug and peek the result
    Return:
        image (PIL image shape: (h, w, channel) dtype: uint8): 
            The image where bounding box, the class names as well as the scoreshould are drawed on.
    """
    
    #load font from yad2k lib
    font = ImageFont.truetype(
            font='yad2k/font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = 3
    
    # Set default color for white
    if not colors:
        colors=(0,0,0)
        
    for i, c in reversed(list(enumerate(out_classes))):
        
        predicted_class = c
        box = out_boxes[i]

        label = '{}'.format(predicted_class)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        
        # Extract four points of the bounding box

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        
        
        if debug:
            print(label, (left, top), (right, bottom))
        
        # Define text position using label_size
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # Draw bounding box and text
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors)
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors)
        draw.text(text_origin, label, fill=(255, 255, 255), font=font)
        del draw
    return image

def detect_and_track(sess, 
                      yolo_model,
                      frame, 
                      parameter,
                      tracker_list = [ [], [], [] ],
                      memory={},
                      debug = False):
    """
    Extract detected object from the image using pretrained yolo model, 
    then use simple track by detection method to track object
    Arguments:
        sess (tf sesssion):
            The tensorflow session  
        
        yolo_model (keras model): 
            The pretrained keras yolo model
            
        frame (PIL image):
            The input PIL image
            
        parameter (dict):
            A dictionary contain following parameters:
                input_image_shape (tf placeholder shape: (2, ) dtype: int32): 
                    The shape of the image in the form of [frame.size[1], frame.size[0]]
                boxes (tf tensor shape: (2, ) dtype: int32):
                    The tensor ot keep the output boxes after the detection of yolo model
                scores (tf tensor shape: (2, ) dtype: int32):
                    The tensor ot keep the output scores for each boxes after the detection of yolo model
                classes (tf tensor shape: (2, ) dtype: int32):
                    The tensor ot keep the output classes for each boxes after the detection of yolo model
                model_image_size (tuple dtype: int):
                    The input size for the pretrained yolo model
                class_names (list dtype: string):
                    The class_names of coco datasets
                dis_threshold (int):
                    The distance threshold to pair up two bounding boxes
                    
        tracker_list (list dtype: list):
            tracker_list[0] (tuple int32)： the stored bounding box
            tracker_list[1] (tuple int32)： the stored out_classes
            tracker_list[2] (tuple int32)： the stored identity
            
        memory (dict key: str val: int):
            The memory for different classes
        
        debug (bool): 
            The debug flag for me to debug and peek the result
    Return:
        bbox (list of list):
            The list of output bounding box
            
        identity (list of string):
            The list of output id for the object
    """
    # Preprocessing the PIL image data
    # resize -> to numpy -> normalize -> expend dimenstion to (1, w, h, channel)
    resized_image = frame.resize(
                tuple(reversed(parameter["model_image_size"])), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)[:,:,:,0:3]
    input_shape = [frame.size[1], frame.size[0]]
    
        
    #feed model to get output parameters
    out_boxes, out_scores, out_classes = sess.run(
            [parameter["boxes"], parameter["scores"], parameter["classes"]],
            feed_dict={
                yolo_model.input: image_data,
                parameter["input_image_shape"]: input_shape,
                K.learning_phase(): 0
            })
    if debug:
        print("The shape of resized image is ", image_data.shape)
    '''
    the tracking part start from here
    '''
    #initialize output
    i = 0
    identity, bbox = [], []
    del_dict = {}
    while i < len(tracker_list[0]):
        
        # if object is found by the not detector
        if not tracker_list[1][i] in out_classes:
            del tracker_list[0][i], tracker_list[1][i], tracker_list[2][i]
        # compare tracker and detector result
        else:
            #search detected bbox to find the min distance
            min_dis, min_id = parameter["dis_threshold"], -1
            for j in range(len(out_classes)):
                if out_classes[j] == tracker_list[1][i]:
                    distance = MiddlePointDistance(out_boxes[j], tracker_list[0][i])
                    if min_dis > distance:
                        min_dis, min_id = distance, j
            
            # find good result
            if min_id > -1:
                
                #update tracker_list using detection result
                tracker_list[0][i] = out_boxes[min_id]
                tracker_list[1][i] = out_classes[min_id]
                
                #delete the corresponding detection result
                del_dict[min_id] = True
                
                #append info to identity and bbox
                identity.append(tracker_list[2][i])
                bbox.append(tracker_list[0][i])
                i+=1
            else:
                del tracker_list[0][i], tracker_list[1][i], tracker_list[2][i]
        
    #initialize the rest detection result as new trackers
    for i in range(len(out_boxes)):
             
        if i in del_dict:
            continue
        
        #append new tracker inforamtion to tracker list  
        tracker_list[0].append(out_boxes[i])
        tracker_list[1].append(out_classes[i])
        
        c_name = parameter["class_names"][out_classes[i]]
        memory[c_name]+=1
        tracker_list[2].append(c_name+"_"+str(memory[c_name]))
        
        #append info to 
        identity.append(tracker_list[2][-1])
        bbox.append(tracker_list[0][-1])
        
    if debug:
        print(tracker_list[0])
        print(tracker_list[1])
        print(tracker_list[2])
        
    return bbox, identity 