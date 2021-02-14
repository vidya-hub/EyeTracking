
import cv2
import numpy as np


def empty(a):
    #
    pass


def stackimage(scale, imgarray):
    rows = len(imgarray)
    cols = len(imgarray[0])
    rowsAvailable = isinstance(imgarray[0], list)
    width = imgarray[0][0].shape[1]
    height = imgarray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgarray[x][y].shape[:2] == imgarray[0][0].shape[:2]:
                    imgarray[x][y] = cv2.resize(
                        imgarray[x][y], (0, 0), None, scale, scale)
                else:
                    imgarray[x][y] = cv2.resize(
                        imgarray[x][y], (imgarray[0][0].shape[1], imgarray[0][0].shape[0]), None, scale, scale)
                if len(imgarray[x][y].shape) == 2:
                    imgarray[x][y] = cv2.cvtColor(
                        imgarray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgarray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgarray[x].shape[:2] == imgarray[0].shape[:2]:
                imgarray[x] = cv2.resize(
                    imgarray[x], (0, 0), None, scale, scale)
            else:
                imgarray[x] = cv2.resize(
                    imgarray[x], (imgarray[0].shape[1], imgarray[0].shape[0]), None, scale, scale)
            if len(imgarray[x].shape) == 2:
                imgarray[x] = cv2.cvtColor(imgarray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgarray)
        ver = hor
    return ver


cap = cv2.VideoCapture("video.mp4")
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 19, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 110, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 240, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 153, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

eyecascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)


def findrect(coordinates):
    (x1, y1, w1, h1) = coordinates[0]
    (x2, y2, w2, h2) = coordinates[1]
    (x, y, w, h) = (x1, y1, abs(x2+w2-x1), h2)
    return (x, y, w, h)


while True:
    # _, img = cap.read()
    _, result = cap.read()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    eyes = eyecascade.detectMultiScale(gray, 1.3, 5)
    if len(eyes) == 2:
        (x, y, w, h) = findrect(eyes)
        eyesimage = result[y:y+h, x:x+w]
        imgHSV = cv2.cvtColor(eyesimage, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        print(h_min, h_max, s_min, s_max, v_min, v_max)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(eyesimage, eyesimage, mask=mask)

        cv2.imshow("Original", eyesimage)
        cv2.imshow("HSV", imgHSV)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", imgResult)

        # imgStack = stackimage(
        #     0.6, ([eyesimage[400:1500, 70:1145], imgHSV[400:1500, 70:1145]],
        #           [mask[400:1500, 70:1145], imgResult[400:1500, 70:1145]]))
        # cv2.imshow("Stacked Images", imgStack)

        if cv2.waitKey(1) == 27:
            break
cv2.destroyAllWindows()
