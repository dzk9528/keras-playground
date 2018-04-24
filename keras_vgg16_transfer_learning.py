# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:09:58 2018

A practice for keras transfer learning,
small but fun
@author: zhang
"""
import numpy as np
import os
import time
import cv2
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Dense,Flatten
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal
from random import shuffle


def get_dataset():
    """
    obtain dataset from folder and separate dataset into 
    training dataset and vaidation data.
    
    the dataset is available to download on kaggle:
    https://www.kaggle.com/iamprateek/vehicle-images-gti
    
    Returns:
        train_x: numpy array, the trainning images
        train_y: numpy array, the label of trainning images
        val_x: numpy array, the validation images
        val_y: numpy array, the validation images
        train_size: int, 
    """
    path = 'C:/Users/zhang/dp/VehicleImage/'
    p1=path+'vehicles/'
    p0=path+'non-vehicles/'
    imageData=[]
    label=[]
    
    #read negative samples
    for subPath in ['Far/','Left/','MiddleClose/','Right/']:
        for filename in os.listdir(p0+subPath):
            img_path=p0+subPath+filename
            im=cv2.imread(img_path)
            imageData.append(im)
            label.append(0)
    
    #read positive samples
    for subPath in ['Far/','Left/','MiddleClose/','Right/']:
        for filename in os.listdir(p1+subPath):
            img_path=p1+subPath+filename
            im=cv2.imread(img_path)
            imageData.append(im)
            label.append(1)
    
    #generate permutation
    p = np.random.permutation(len(label)) 
    
    #set up the size of training samples
    train_size = int(len(imageData)*0.85)
    val_size = len(imageData)-train_size
    #normalize image dataset
    imageData=np.array(imageData)[p]/255.5-0.5
    
    #change label to proper data type
    label=np.array(label).reshape([len(label),1])[p]*1.0
    
    print("Now the dataset is separate into two parts")
    print(train_size," data for training")
    print(val_size," data for validation")
    
    #separate data into two part
    train_x=imageData[:train_size,:,:]
    train_y=label[:train_size,:]
    val_x=imageData[train_size:,:,:]
    val_y=label[train_size:,:]

    return train_x,train_y,val_x,val_y,train_size

def compute_acc(y,y_pred):
    """
    compute result giving label and prediction
    Args:
        y: numpy array, the labels of the sample
        y_pred: numpy array, the prediction
    Returns:
        acc: float32, the accuracy round to 5
    """
    acc=K.eval(K.mean(K.equal(K.cast(y,dtype='float32'), K.round(y_pred))))  
    return round(acc,5)

def plot_acc(epoch,train_acc, val_acc):
    """
    plot the training accuracy and validation accuracy change as epoch increases
    Args:
        epoch: list, the range from 0 to epoch
        train_acc: list, the list of training accuracy
        val_acc: list, the list of validation accuracy
    """
    fig,ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    
    #put accuracy into plot
    tr_l=ax1.plot(epoch, train_acc, 'r-', label='batch training error')
    te_l=ax1.plot(epoch, val_acc, 'b-', label='validation error')
    tr_l_p=ax1.plot(epoch, train_acc, 'r*')
    te_l_p=ax1.plot(epoch, val_acc, 'b*')
    
    #set up y axis 
    ax1.set_ylabel('accuracy', color='orange')
    ax1.tick_params('y', colors='orange')
    
    #combine 2 plots together
    lns =tr_l+te_l+tr_l_p+te_l_p
    labs=[l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    fig.tight_layout()
    
    plt.show()
     
def main():
    start=time.time()
    print("Start to read data...")
    
    #get training data and validation data from folder
    train_x,train_y,val_x,val_y,train_size=get_dataset()
    
    print("finished in %s seconds"%(round(time.time()-start,4)))
    
    #load basic model
    base_model = VGG16(weights="imagenet",include_top=False,input_shape=(64,64,3))
    
    #keep the cnn feature of vgg16
    for layer in base_model.layers:
        layer.trainable = False
        
    #set up initiable weights for classifier
    init=TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
    
    #flatten output of base model
    x = Flatten()(base_model.output)
    
    # add fully connect layer
    x = Dense(1024, kernel_initializer=init, activation='relu')(x)
    
    # add classifier
    predictions = Dense(1, kernel_initializer=init,activation='sigmoid')(x)
    
    #catencate the model togrther
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print(model.summary())
    
    # set up adam grdaient
    adam=Adam(lr=0.00008, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    #compile model
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #basical paramenter setting
    eps=20
    val_acc=[]
    train_acc=[]
    max_acc=0.9
    
    #generate initial prediction of batch training data and validation data
    train_y_pred=model.predict(train_x[:128])
    train_acc.append(compute_acc(train_y[:128],train_y_pred))
    val_y_pred=model.predict(val_x)
    val_acc.append(compute_acc(val_y,val_y_pred))
    
    
    print("batch training accuracy is %s for epoch %s" %(train_acc[-1], 0))
    print("validation accuracy is %s for epoch %s" %(val_acc[-1], 0))
    
    for i in range(eps):
        #fit model
        model.fit(train_x, train_y, epochs=1, batch_size=128)
        
        #generate predictions of batch training data and validation data
        train_y_pred=model.predict(train_x[:128])
        train_acc.append(compute_acc(train_y[:128],train_y_pred))
        val_y_pred=model.predict(val_x)
        val_acc.append(compute_acc(val_y,val_y_pred))
        
        print("batch training accuracy is %s for epoch %s" %(train_acc[-1], i+1))
        print("validation accuracy is %s for epoch %s" %(val_acc[-1], i+1))
        
        #save the model if 
        if val_acc and val_acc[-1]>max_acc:
            max_acc=val_acc[-1]
            model.save('vgg16_transfer_learning_model.h5')
    if max_acc>0.9:
        print("Training completed, the max accuracy is %s"%(max_acc))
        plot_acc(range(0,len(train_acc)),train_acc, val_acc)
    
    return
if __name__=="__main__":
    main()
