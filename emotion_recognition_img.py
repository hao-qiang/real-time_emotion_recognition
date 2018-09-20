import os
import sys
import cv2
import numpy as np

from keras.models import *
from keras.layers import *
from keras.applications import *
from models import emotion_recognition_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
input_path = sys.argv[1]
img = np.zeros((1, 128, 128, 3), np.float32)
img[0] = cv2.resize(cv2.imread(input_path), (128,128))[:,:,::-1] / 255.

model = emotion_recognition_model('weights/mobilenet_0.4379_0.8605.hdf5')
pred = model.predict(img)

print('Predict result:', labels[np.argmax(pred[0])])