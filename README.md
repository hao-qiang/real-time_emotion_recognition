# real-time emotion recognition
A real-time emotion recognition project which can run on CPU.

![](https://github.com/hao-qiang/real-time_emotion_recognition/blob/master/emotion_detection.png)

## Requirements

- python 3.6
- opencv 3.4.1
- tensorflow 1.5
- keras 2.1.5

## Running

Using pc camera to do real-time emotion recognition.

```
python emotion_recognition_cam.py
```

## Pipline

1. Using [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment) model to detecte face and its landmarks.
2. Using [RAF-DB](http://www.whdeng.cn/raf/model1.html) emotion dateset(aligned) to train a [mobilenet](https://arxiv.org/abs/1704.04861) model (test acc=86.05%).
