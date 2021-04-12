import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import concatenate, Lambda, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import keras.backend as K
from keras.datasets import cifar10
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import itertools
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import pydot
from IPython.display import SVG
#--------------------------------------------------------------------------------------------------------------------------------------------------
#Import images
def getimages(direct):
    im=[]
    label=[]
    lab=0
    for labels in os.listdir(direct):
        #Label corresponding to folder
        if labels=="buildings":
            lab=0
        if labels=="forest":
            lab=1
        if labels=="glacier":
            lab=2
        if labels=="mountain":
            lab=3
        if labels=="sea":
            lab=4
        if labels=="street":
            lab=5
        for image in os.listdir(direct+r'/'+labels):
            #Read images
            imag=cv2.imread(direct+r'/'+labels+r'/'+image)
            imag=cv2.resize(imag,(150,150))
            im.append(imag)
            label.append(lab)
    #Suffle all images
    return shuffle(im, label)

#Obtain both train and test images
im,label=getimages("archive\seg_train\seg_train")      
im1,label1=getimages("archive\seg_test\seg_test") 
img=im+im1
lab=label+label1

#Convert to numpy array
Images=np.array(img)
Labels=np.array(lab)
classes=6

#Train-test-validation split
#Split into train, validation and test
x1, xtest, y1,ytest=train_test_split(Images, Labels, random_state=0)
xtrain, xvalidation, ytrain, yvalidation=train_test_split(x1, y1, random_state=0)
#Prepare x
xtrain = xtrain.astype('float32')
xvalidation = xvalidation.astype('float32')
xtest=xtest.astype('float32')
xtrain /= 255
xvalidation /= 255
xtest /= 255
#Prepare y
ytrain0 = keras.utils.to_categorical(ytrain, classes)
yvalidation0 = keras.utils.to_categorical(yvalidation, classes)
ytest0=keras.utils.to_categorical(ytest, classes)
#--------------------------------------------------------------------------------------------------------------------------------
#Get models
# Small VGG-like model
def simpleVGG(im_Input, num_classes, name="vgg"):
    name = [name+str(i) for i in range(12)]
    
    # convolution and max pooling layers
    vgg = Conv2D(32, (3, 3), padding='same', activation='relu', name=name[0])(im_Input)
    vgg = Conv2D(32, (3, 3), padding='same', activation='relu', name=name[1])(vgg)
    vgg = MaxPooling2D(pool_size=(2,2), name=name[2])(vgg)
    vgg = Dropout(0.25, name=name[3])(vgg)
    vgg = Conv2D(64, (3, 3), padding='same', activation='relu', name=name[4])(vgg)
    vgg = Conv2D(64, (3, 3), padding='same', activation='relu', name=name[5])(vgg)
    vgg = MaxPooling2D(pool_size=(2,2), name=name[6])(vgg)
    vgg = Dropout(0.25, name=name[7])(vgg)

    # classification layers
    vgg = Flatten(name=name[8])(vgg)
    vgg = Dense(512, activation='relu', name=name[9])(vgg)
    vgg = Dropout(0.5, name=name[10])(vgg)
    vgg = Dense(num_classes, activation='softmax', name=name[11])(vgg)
    return vgg
# get the newest model file within a directory
def getNewestModel(model, dirname):
    from glob import glob
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    if len(files) == 0:
        return model
    else:
        newestModel = sorted(files, key=lambda files: files[1])[-1]
        model.load_weights(newestModel[0])
        return model
        
#Initialize models
im_Input = Input(shape=(xtrain.shape[1:]), name="input")
#6 class
baseVGG = simpleVGG(im_Input, 6, "base")
baseModel = Model(im_Input, baseVGG)
baseModel.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#3 class
threeVGG = simpleVGG(im_Input, 2, "three")
threeModel = Model(im_Input, threeVGG)
threeModel.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#Nature
natVGG = simpleVGG(im_Input, 6, "nature")
natModel = Model(im_Input, natVGG)
natModel.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#man-made
manVGG = simpleVGG(im_Input, 6, "manmade")
manModel = Model(im_Input, manVGG)
manModel.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

#Retrieve models for every block of the network
#6 class classifier
baseSaveDir = "./base_Im_1/"
baseModel = getNewestModel(baseModel, baseSaveDir)
#2class classifier
ThreeSaveDir = "./2_classes_1/"
threeModel = getNewestModel(threeModel, ThreeSaveDir)
#Man-made
manSaveDir = "./man_Im/"
manModel = getNewestModel(manModel, manSaveDir)
#Nature classifier
natSaveDir = "./nat_Im/"
natModel = getNewestModel(natModel, natSaveDir)
#--------------------------------------------------------------------------------------------------------------------------
#Put all together with sub-gates
for l in baseModel.layers:
    l.trainable = False
for l in threeModel.layers:
    l.trainable = False
for l in manModel.layers:
    l.trainable = False
for l in natModel.layers:
    l.trainable = False
# define sub-Gate network, for the second gating network layer
def subGate(Input, orig_classes, numExperts, name="subGate"):
    name = [name+str(i) for i in range(5)]
    subgate = Flatten(name=name[0])(Input)
    subgate = Dense(512, activation='softmax', name=name[1])(subgate)
    subgate = Dropout(0.5, name=name[2])(subgate)
    subgate = Dense(orig_classes*numExperts, activation='softmax', name=name[3])(subgate)
    subgate = Reshape((orig_classes, numExperts), name=name[4])(subgate)
    return subgate
# the artificial gating network
manGate = subGate(im_Input, classes, 2, "manExpertGate")
# the natural gating network
natureGate = subGate(im_Input, classes, 2, "natureExpertGate")
# Define functioning of subgates
def subGateLambda(base, expert, subgate):
    output = Lambda(lambda gx: (gx[0]*gx[2][:,:,0]) + (gx[1]*gx[2][:,:,1]), output_shape=(classes,))([base, expert, subgate])
    return output
#Final output version 1: with base classifier and subgates
output = Lambda(lambda gx: K.switch(gx[1][:,0] > gx[1][:,1], 
                                    subGateLambda(gx[0], gx[2], gx[4]), 
                                    subGateLambda(gx[0], gx[3], gx[5])),
                output_shape=(classes,))([baseVGG, threeVGG, manVGG, natVGG, manGate, natureGate])
#Final output without base classifier
output1=Lambda(lambda gx: K.switch(gx[1][:,0] > gx[1][:,1], 
                                    gx[2], 
                                    gx[3]),
                output_shape=(classes,))([baseVGG, threeVGG, manVGG, natVGG])
model = Model(im_Input, output1)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
#--------------------------------------------------------------------------------------------------------------------------------
# evaluate
trainScore = model.evaluate(xtrain, ytrain0)
valScore = model.evaluate(xvalidation, yvalidation0)
testScore = model.evaluate(xtest, ytest0)
print(trainScore)
print(valScore)
print(testScore)





