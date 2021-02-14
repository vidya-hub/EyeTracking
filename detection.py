import cv2
import numpy as np

eyecascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)


def empty(a):
    #
    pass


def findrect(coordinates):
    (x1, y1, w1, h1) = coordinates[0]
    (x2, y2, w2, h2) = coordinates[1]
    (x, y, w, h) = (x1, y1, abs(x2+w2-x1), h2)
    return (x, y, w, h)


# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars", 640, 240)
# cv2.createTrackbar("thresh Min", "TrackBars", 0, 179, empty)

while True:
    _, result = cap.read()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    eyes = eyecascade.detectMultiScale(gray, 1.3, 5)

    if len(eyes) == 2:
        (x, y, w, h) = findrect(eyes)
        eyesimage = result[y:y+h, x+5:x+5+w]
        eyesgray = cv2.cvtColor(eyesimage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(eyesgray, (5, 5), 0)
        # t_min = cv2.getTrackbarPos("thresh Min", "TrackBars")
        _, img = cv2.threshold(blur, 37, 255, cv2.THRESH_BINARY)
        cv2.imshow("eyes", img)

    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
