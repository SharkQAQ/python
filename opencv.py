import cv2
from pyzbar import pyzbar


def decodedisplay(image, img):
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        # 提取条形码的边界框的位置
        # 画出图像中条形码的边界框
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)

        # 条形码数据为字节对象，所以如果我们想在输出图像上
        # 画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

        # 绘出图像上条形码的数据和条形码类型
        # text = "{} ({})".format(barcodeData, barcodeType)
        # cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #             .5, (0, 0, 125), 2)

        # 向终端打印条形码数据和条形码类型
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
    return img


def detect():
    camera = cv2.VideoCapture(0)
    while True:
        # 读取当前帧
        ret, frame = camera.read()
        # 转为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im = decodedisplay(gray, frame)
        if ord('q') == cv2.waitKey(10):
            break
        cv2.imshow("camera", im)
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect()
