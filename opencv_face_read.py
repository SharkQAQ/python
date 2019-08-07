import cv2


def read():
    idnum = 0
    names = dict()
    with open('name', 'r+') as op:
        names = eval(op.read())
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('face_trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # idnum = 0
    # names = []  # 照片id=0的用户姓名排在第一位 id=1的排在第二位 以此类推，可存数据库

    cam = cv2.VideoCapture(0)
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH))
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])  # 获取人脸预测结果 编号和不符合比例
            if confidence < 100:
                name = names[str(idnum)]
                confidence1 = "{0}%".format(round(100 - confidence))
            else:
                name = "unknown"
                confidence1 = "{0}%".format(round(100 - confidence))
            cv2.putText(img, str(name), (x+5, y-5), font, 1, (0, 0, 255), 1)
            if confidence > 50:  # 不符合比例大于一半就变红色
                cv2.putText(img, str(confidence1), (x+5, y+h-5), font, 1, (0, 0, 255), 1)
            else:
                cv2.putText(img, str(confidence1), (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)

        cv2.imshow('camera', img)
        k = cv2.waitKey(10)
        if k == 27:
            break
    print(names)
    cam.release()
    cv2.destroyAllWindows()
    return
