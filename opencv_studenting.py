import numpy as np
from PIL import Image
import os
import cv2
def student():
    # 人脸数据路径
    path = 'Facedata'

    # pip install opencv-contrib-python
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # opencv自带学习方式
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')   # 读取图片库的图片为黑白8像素
            img_numpy = np.array(PIL_img, 'uint8')  # 将图片存到array数组中
            id = int(os.path.split(imagePath)[-1].split(".")[1])  # 获取每张图片的名称(代表的信息)
            faces = detector.detectMultiScale(img_numpy)  # 检测人脸
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x: x + w])  # 获取人脸数据数组
                ids.append(id)  # 获取人脸id
        return faceSamples, ids
    print('脸部信息获取完成，计算机正在训练脸部信息中. 请等待 ..')
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))  # 将人脸数据进行处理学习

    recognizer.write(r'face_trainer\trainer.yml')  # 将学习好的人脸数据保存
    print("{0} 个脸部训练完成. 开始识别".format(len(np.unique(ids))))
    return