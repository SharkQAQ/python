import cv2
names = dict()


def getface():
    # 调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_id = input('\n 输入你的id(数字):')
    face_name = input('\n 请输入你的名字:')
    name = {face_id: face_name}
    names.update(name)
    with open('name', 'w') as op:
        op.write(str(names))
    print('\n 正在采集脸部信息1000张，请将脸部对准摄像头 ...')
    count = 0

    while True:
        # 从摄像头读取图片
        flag, img = cap.read()
        # 转为灰度图片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # 画边框
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            count += 1
        # 保存图像
            cv2.imwrite("Facedata/User." + str(face_id) + '.' + str(count) + '.jpg', gray[y: y + h, x: x + w])
            cv2.putText(img, str(count), (x + 5, y - 5), font, 1, (0, 0, 255), 1)
        cv2.imshow('image', img)
    # 保持画面的持续。
        k = cv2.waitKey(1)
        if k == 27:   # 通过esc键退出摄像
            break

        elif count >= 1000:  # 得到1000个样本后退出摄像
            break

    # 关闭摄像头 释放read
    cap.release()
    cv2.destroyAllWindows()
    return
