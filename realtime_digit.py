import numpy as np
import cv2
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.preprocessing import image


width = 640
height = 480



with open('/Users/kandagadlaashokkumar/Desktop/neuralnetworks/Digit Classification/digit_model.json', 'r') as json_file:
    json_savedModel= json_file.read()
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights('/Users/kandagadlaashokkumar/Desktop/neuralnetworks/Digit Classification/digit_weights.h5')


def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img




imageOrg = cv2.imread("/Users/kandagadlaashokkumar/Desktop/6(2).jpg")
gray_img = cv2.cvtColor(imageOrg,cv2.COLOR_BGR2GRAY)
roi_gray = cv2.resize(gray_img,(32,32))
img_pixels = image.img_to_array(roi_gray)
img_pixels = np.expand_dims(img_pixels,axis = 0)
img_pixels/=255
cv2.imshow("org",imageOrg)
classIndex = int(model.predict_classes(img_pixels))
predictions = model.predict(img_pixels)
probval = np.amax(predictions)
print(classIndex,probval)
cv2.waitKey(0)

