# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import glob
from os import listdir
from os.path import isfile, join
from commonfunctions import *
from Segmentation import *
import imageio as iio
from skimage import filters
from skimage.color import rgb2gray  # only needed for incorrectly saved images
from skimage.measure import regionprops
import collections
import timeit
import datetime
import time
#***********************************************************************************


#*******Intilize Neural Network Model with 2 hidden Layers********
def InilizedModel():
    # define the architecture of the network
    model = Sequential()
    model.add(Dense(40, input_dim=20, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(29))
    model.add(Activation("softmax"))
    # train the model using SGD
    sgd = SGD(lr=0.01)
    model.compile(loss="binary_crossentropy", optimizer=sgd,metrics=["accuracy"])
    return model
#***********************************************************************************   

#*******Prepare Data and Labels before entering the model********
def PrepareLabelsData(data,labels):
    # encode the labels, converting them from strings to integers and normalize data
    
    labels = list(map(int, labels))
    #print("Labels: ",labels)
    
    data = np.asarray(data)
    labels = np_utils.to_categorical(labels, 29)
    #print(labels)
    return data,labels
#***********************************************************************************   


#*******Train Model with 3 epochs and save weights*******
def TrainModel(model,data,labels,flag):
    model.fit(data, labels, epochs=3, batch_size=128,verbose=1)
    model.save("TrainedModelFinal.h5")
    flag=True
    return model,flag
#***********************************************************************************   

#******Load the Saved Model*******
def LoadModel(modelname):
   
    # Returns a compiled model identical to the previous one
    model = load_model(modelname)
    return model
#***********************************************************************************   


#******Evalute Model on the 20% of the dataset*******
def EvaluteModel(model,data,labels):
    # show the accuracy on the testing set
    #print("[INFO] evaluating on testing set...")
    mypath = r'lettersfinal/validation/'
    Number_Of_Files = len([ f for f in listdir(mypath) if isfile(join(mypath,f)) ])
    #print(Number_Of_Files)
    gen =  glob.iglob(mypath + "*.png")
    for i in range(Number_Of_Files):
        py = next(gen)
        underscore=py.split("_")
        #print(lbl[1])
        lbl=underscore[1].split(".")
        labels.append(lbl[0])
        input_image = cv2.imread(py)
        #show_images([input_image])
        # print(labels)
        #print(py)
        gray = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
        data.append(image_to_feature_vector(gray))
    data, labels = PrepareLabelsData(data, labels)
    (loss, accuracy) = model.evaluate(data, labels,batch_size=128, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
    return loss,accuracy
#***********************************************************************************   

#******Predict input letter image*******
def PredictImage(model,Predictimage):
    #Predictimage = cv2.imread(imagepath)
    #gray = cv2.cvtColor(Predictimage, cv2.COLOR_BGR2GRAY)
    features = image_to_feature_vector(Predictimage)
    features = np.asarray([features])
    probs = model.predict(features)[0]
    #print(probs)
    prediction = probs.argmax(axis=0)
    #print(prediction)
    return prediction
#***********************************************************************************   

#******Write the output letters in words in text file*******
def WriteOutput(wordlist,filehandle):
    for listitem in wordlist:
        filehandle.write('%s' % listitem)
    filehandle.write(' ')
#***********************************************************************************   

#******arabic dictionary constructor to get value predicted from the model*******
def ConstructArabicDict():
    ArabicDictionary = {}
    ArabicDictionary['ا'] = 1
    ArabicDictionary['ب'] = 2
    ArabicDictionary['ت'] = 3
    ArabicDictionary['ث'] = 4
    ArabicDictionary['ج'] = 5
    ArabicDictionary['ح'] = 6
    ArabicDictionary['خ'] = 7
    ArabicDictionary['د'] = 8
    ArabicDictionary['ز'] = 9
    ArabicDictionary['ر'] = 10
    ArabicDictionary['ز'] = 11
    ArabicDictionary['س'] = 12
    ArabicDictionary['ش'] = 13
    ArabicDictionary['ص'] = 14
    ArabicDictionary['ض'] = 15
    ArabicDictionary['ط'] = 16
    ArabicDictionary['ظ'] = 17
    ArabicDictionary['ع'] = 18
    ArabicDictionary['غ'] = 19
    ArabicDictionary['ف'] = 20
    ArabicDictionary['ق'] = 21
    ArabicDictionary['ك'] = 22
    ArabicDictionary['ل'] = 23
    ArabicDictionary['م'] = 24
    ArabicDictionary['ن'] = 25
    ArabicDictionary['ه'] = 26
    ArabicDictionary['و'] = 27
    ArabicDictionary['ي'] = 28
    ArabicDictionary['لا'] = 29
    return ArabicDictionary
#***********************************************************************************   

#******get actual letter predicted from the model*******
def GetLetterFromPrediction(ArabicDictionary, Prediction):
    key_list = list(ArabicDictionary.keys())
    val_list = list(ArabicDictionary.values())

    return key_list[val_list.index(Prediction)]
#***********************************************************************************   

#******see if the letter contain any holes or not*******
# def WithHoles(im,gray):
#     _,contours,_ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         print (area)
#         if area < 20:
#             cv2.drawContours(im, [cnt], 0, (255, 0, 0), 2)
#     show_images([im])
#***********************************************************************************   

#******get center of mass for the letter image*******
def CenterofMass(resized):
    threshold_value = filters.threshold_otsu(resized)
    labeled_foreground = (resized > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, resized)
    center_of_mass = properties[0].centroid
    weighted_center_of_mass = properties[0].weighted_centroid
    com=[]
    com.append(weighted_center_of_mass)
    return com
#***********************************************************************************   

#******get black on white and black on black ratios in 4 regions*******
def RatioRegions(threshed_img):
    RatioList = []
    resized1 = threshed_img[0:25,0:25]/255
    resized2 = threshed_img[25:50,0:25]/255
    resized3 = threshed_img[0:25,25:50]/255
    resized4 = threshed_img[25:50,25:50]/255

    bw11 = (625-np.sum(resized1))/(np.sum(resized1)+1)
    bw22 = (625-np.sum(resized2))/(np.sum(resized2)+1)
    bw33 = (625-np.sum(resized3))/(np.sum(resized3)+1)
    bw44 = (625-np.sum(resized4))/(np.sum(resized4)+1)

    bb12 = (625-np.sum(resized1))/(625-np.sum(resized2))
    bb34 = (625-np.sum(resized3))/(625-np.sum(resized4))
    bb13 = (625-np.sum(resized1))/(625-np.sum(resized3))
    bb24 = (625-np.sum(resized2))/(625-np.sum(resized4))
    bb14 = (625-np.sum(resized1))/(625-np.sum(resized4))
    bb23 = (625-np.sum(resized2))/(625-np.sum(resized3))

    RatioList.append(bw11)
    RatioList.append(bw22)
    RatioList.append(bw33)
    RatioList.append(bw44)
    RatioList.append(bb12)
    RatioList.append(bb34)
    RatioList.append(bb13)
    RatioList.append(bb24)
    RatioList.append(bb14)
    RatioList.append(bb23)
    #RatioList=np.asarray(RatioList)
    return RatioList
#***********************************************************************************   

#******get labels of connected components*******
def ConnectedComponents(thresh):
    connectivity = 8
    out=[]
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    out.append(output[0])
    #out=np.asarray(out)
    return out
#***********************************************************************************   

#******Get Vertical Projection*******
def getVerticalProjectionProfile(image):
    vertical_projection = np.sum(image, axis=0)
    return vertical_projection
#***********************************************************************************   

#******Get Horizontal Projection*******
def getHorizontalProjectionProfile(image):
    horizontal_projection = np.sum(image, axis=1)
    return horizontal_projection
#***********************************************************************************   

#******Black INK Histogram *******
def BlackInkHist(thresh):
    hist = []
    hist.append(getVerticalProjectionProfile(thresh))
    hist.append(getHorizontalProjectionProfile(thresh))
    return hist
#***********************************************************************************   

#******get vector of 7 humoments*******
def HuMoment(image):
    hu=[]
    hu=cv2.HuMoments(cv2.moments(image))
    return hu
#***********************************************************************************   

#******flatten list of lists*******
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
#***********************************************************************************   

#******Feature vector for the letter image*******
def image_to_feature_vector(image):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    feature_vector = []
    resized = cv2.resize(image, (50, 50))
    ret, threshed_img = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    feature_vector.append(HuMoment(resized))
    feature_vector.append(RatioRegions(threshed_img))
    feature_vector.append(ConnectedComponents(threshed_img))
    #feature_vector.append(BlackInkHist(threshed_img))
    feature_vector.append(CenterofMass(threshed_img))
    return np.asarray(flatten(feature_vector)).flatten()
#***********************************************************************************   
    
#******main for training model*******
def main_train():
    data = []
    labels = []
    model=InilizedModel()
    count=0
    TrainedFlag=False

    mypath = r'lettersfinal/training/'
    Number_Of_Files = len([ f for f in listdir(mypath) if isfile(join(mypath,f)) ])
    #print(Number_Of_Files)
    gen =  glob.iglob(mypath + "*.png")
    for i in range(Number_Of_Files):
        count+=1
        if count == 50000 or i==(Number_Of_Files-1):
            print(i)
            if TrainedFlag==True:
                data,labels=PrepareLabelsData(data,labels)
                #print(labels)
                TrainedModel, TrainedFlag = TrainModel(TrainedModel, data, labels, TrainedFlag)
            else:
                data,labels=PrepareLabelsData(data,labels)
                TrainedModel, TrainedFlag = TrainModel(model, data, labels, TrainedFlag)
            data=[]
            labels=[]
            count=0
        py = next(gen)
        underscore=py.split("_")
        #print(lbl[1])
        lbl=underscore[1].split(".")
        labels.append(lbl[0])
        input_image = cv2.imread(py)
        #show_images([input_image])
        # print(labels)
        #print(py)
        gray = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
        data.append(image_to_feature_vector(gray))
#***********************************************************************************   
#train here
#main_train()

#Predict here
# model=LoadModel('TrainedModelFinal.h5')
# pred=PredictImage(model,'lettersfinal/validation/capr1620-49_27.png')

#Evaluate Here
# data = []
# labels = []
# model=LoadModel('TrainedModelFinal.h5')
#loss,accuracy=EvaluteModel(model,data,labels)


#******main for testing and predicting output give image contain text*******
def main_test(thresh):  # to segment the test image
    Arabic_Dict=ConstructArabicDict()
    TrainedModel=load_model("TrainedModelFinal.h5")
    lines = SegmentImg2Lines(thresh)
    filehandle = open('Output.txt', 'w+', encoding='utf-8')
    for line in lines:
        # line = lines[-1]
        words, _ = Segmentline2word(line)
        # words,_ = Segmentline2word(lines[1])
        #show_images([line])

        for word in words:
            # show_images([word])
            BaselineIndex = FindBaselineIndex(word)
            MaxTransitionIndex = FindingMaxTrans(word / 255, BaselineIndex)

            SeparationRegions, MFV = CutPointIdentification(
                word / 255, MaxTransitionIndex)

            ValidSeparationRegions = SeparationRegionFilteration(
                word / 255, SeparationRegions, BaselineIndex, MaxTransitionIndex, MFV)

            ValidSeparationRegions.reverse()  # 3ashan ageb el kelma mn awelha mesh el 3aks
            #
            # word1 = word.copy()
            # for i in range(len(ValidSeparationRegions)):
            #     word1[MaxTransitionIndex, int(ValidSeparationRegions[i].CutIndex)] = 150
            #
            # show_images([word1])

            #WordLettersImage = []
            OutputWord = []
            if len(ValidSeparationRegions) != 0:
                for i in range(len(ValidSeparationRegions) + 1):
                    letter = 0
                    if i == 0:
                        letter = word[:, ValidSeparationRegions[i].CutIndex:]
                    elif i == len(ValidSeparationRegions):
                        letter = word[:, :ValidSeparationRegions[i - 1].CutIndex]
                    else:  # in middle (normal)
                        letter = word[:, ValidSeparationRegions[i].CutIndex: ValidSeparationRegions[i - 1].CutIndex]
                    #show_images([letter])
                    #WordLettersImage.append(letter)
                    Prediction=PredictImage(TrainedModel,letter)
                    OutputLetter=GetLetterFromPrediction(Arabic_Dict,Prediction)
                    OutputWord.append(OutputLetter)

                WriteOutput(OutputWord,filehandle)
#***********************************************************************************   


#******Calling the main test and write time elapsed for performance*******
img = cv2.imread('capr27.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
start = timeit.default_timer()
S = time.time()
main_test(thresh)
stop = timeit.default_timer()
ElapsedTime = time.time() - S
#print('Time: ', stop - start)
#print('elapsed:  ', time.time() - S)
filehandle = open('time.txt', 'w+')
filehandle.write('%s' % ElapsedTime)
#***********************************End of Code************************************************   