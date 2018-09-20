import os
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
import mtcnn_detect_face
from models import emotion_recognition_model, create_mtcnn, get_src_landmarks, process_mtcnn_bbox
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def cal_angle(leftEyeCenter, rightEyeCenter):
    dY = leftEyeCenter[1] - rightEyeCenter[1]
    dX = leftEyeCenter[0] - rightEyeCenter[0]
    return np.degrees(np.arctan2(dY, dX))

def rotation(img, angle):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))
    
    
WEIGHTS_PATH = "weights/"

sess = K.get_session()
with sess.as_default():
    global pnet, rnet, onet 
    pnet, rnet, onet = create_mtcnn(sess, WEIGHTS_PATH)

pnet = K.function([pnet.layers['data']],[pnet.layers['conv4-2'], pnet.layers['prob1']])
rnet = K.function([rnet.layers['data']],[rnet.layers['conv5-2'], rnet.layers['prob1']])
onet = K.function([onet.layers['data']],[onet.layers['conv6-2'], onet.layers['conv6-3'], onet.layers['prob1']])

er_model = emotion_recognition_model('weights/mobilenet_0.4379_0.8605.hdf5')

labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
face_input = np.zeros((1, 128, 128, 3))

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    input_img = img.copy()
    faces, pnts = mtcnn_detect_face.detect_face(input_img, pnet, rnet, onet)
    faces = process_mtcnn_bbox(faces, input_img.shape)
    figure_right = np.zeros((input_img.shape[0], 140, 3), np.uint8)
    
    biggest_face_idx = -1
    biggest_area = 0
    faces_boxes = []
    i = 0
    for (y0, x1, y1, x0, conf_score) in faces:
        if y0>0 and x1>0 and y1>0 and x0>0:
            faces_boxes.append((y0, x1, y1, x0))
            if biggest_area < (y1-y0)*(x1-x0):
                biggest_area = (y1-y0)*(x1-x0)
                biggest_face_idx = i
            i += 1
            
    for idx, (y0, x1, y1, x0) in enumerate(faces_boxes):
        if int(y0)-30>0 and int(x0)-30>0: # rotate the face
            face_img = img[int(y0)-30:int(y1)+30, int(x0)-30:int(x1)+30, :].copy()
            src_landmarks = get_src_landmarks(y0, y1, x0, x1, pnts)
            #for point in src_landmarks:
            #    cv2.circle(face_img,(point[1]+20,point[0]+20), 2, (0,0,255), -1)
            angle = cal_angle((src_landmarks[1][1], src_landmarks[1][0]), (src_landmarks[0][1], src_landmarks[0][0]))
            face_img = rotation(face_img, angle)
            face_img = face_img[30:face_img.shape[0]-30, 30:face_img.shape[1]-30, :]
        else:
            face_img = img[int(y0):int(y1), int(x0):int(x1), :].copy()
        face_input[0] = cv2.resize(face_img, (128, 128))[:,:,::-1] / 255.
        pred_prob = er_model.predict(face_input)
        pred_label = labels[np.argmax(pred_prob[0])]
        
        if idx==biggest_face_idx: # show the biggest face and its emotion degrees
            figure_right[20:120, 20:120, :] = cv2.resize(face_img, (100, 100))
            for i, prob in enumerate(pred_prob[0]):
                figure_right[160+i*40:190+i*40, 20:20+int(100*prob), :] = (0,0,255)
                cv2.putText(figure_right, labels[i], (20+5, 160+i*40+30-6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.rectangle(input_img,(int(x0),int(y0)),(int(x1),int(y1)),(0,0,255),2)
        cv2.rectangle(input_img, (int(x0), int(y1)), (int(x1), int(y1)+32), (0, 0, 255), cv2.FILLED)
        cv2.putText(input_img, pred_label, (int(x0)+6, int(y1)+32-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('img', np.concatenate((input_img, figure_right), axis=1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
