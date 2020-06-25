import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json


path = '/Users/kandagadlaashokkumar/Desktop/neuralnetworks/Digit Classification/myData'
testRatio = 0.2
vaildRatio = 0.2
imageDimensions = (32,32,3)
batchSizeval = 50
epochsVal = 5
stepsperepoch = 700

images = []
classNo = []
myList = os.listdir(path)
print("Total number of classes Detected:",len(myList))
noOfclasses = len(myList)
print("importing CLasses....")

for x in range(0,noOfclasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x)+"/"+y)
        curImg = cv2.resize(curImg,(32,32))
        images.append(curImg)
        classNo.append(x)
    print(x,end= " ")
print(" ")

print("Total number of images in images list:",len(images))
print("Total Ids in classNo lsit",len(classNo))

images = np.array(images)
classNo = np.array(classNo)

x_train,x_test,y_train,y_test = train_test_split(images,classNo,test_size = testRatio)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size = vaildRatio)

print("Total Images:",images.shape)
print("Training images:",x_train.shape)
print("Validation images:",x_validation.shape)
print("Testing Images:",x_test.shape)

numOfSamples = []
for x in range(0,noOfclasses):
    # print(len(np.where(y_train==x)[0]))
    numOfSamples.append(len(np.where(y_train==x)[0]))

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfclasses),numOfSamples)
plt.title("No of training images in each class")
plt.xlabel("class ID")
plt.ylabel("Number of images")
plt.show()

def preProcessing(image):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# print(y_train[30])
# img = x_train[30]
# img = cv2.resize(img,(300,300))
# cv2.imshow("Preprocessed",img)
# cv2.waitKey(0)

x_train = np.array(list(map(preProcessing,x_train)))
x_test = np.array(list(map(preProcessing,x_test)))
x_validation = np.array(list(map(preProcessing,x_validation)))

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)

dataGen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)

dataGen.fit(x_train)

y_train = to_categorical(y_train,noOfclasses)
y_test = to_categorical(y_test,noOfclasses)
y_validation = to_categorical(y_validation,noOfclasses)

def myModel():
    noOfFilters = 16
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape = (imageDimensions[0],imageDimensions[1],1),activation='relu')))
    model.add((Conv2D(noOfFilters,sizeOfFilter1,activation='relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add((Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu')))
    model.add((Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfclasses,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

model = myModel()
print(model.summary())


history = model.fit_generator(dataGen.flow(x_train,y_train,batch_size=batchSizeval),steps_per_epoch = stepsperepoch,epochs = epochsVal,validation_data = (x_validation,y_validation),shuffle=True)




plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(x_test,y_test,verbose =0)
print('Test Score =',score[0])
print("Test Accuracy = ",score[1])



json_model = model.to_json()
with open('digit_model.json', 'w') as json_file:
    json_file.write(json_model)
model.save_weights('digit_weights.h5')
