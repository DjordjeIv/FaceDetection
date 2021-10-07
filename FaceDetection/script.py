import cv2 as cv
import numpy as np
import dlib as dl
import imutils
import matplotlib.pyplot as plt
from imutils import face_utils
from imutils.object_detection import non_max_suppression

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
def points(img, shape):
    i = 1
    for (x, y) in shape:
        if (i > 48 and i < 69) or (i > 40 and i < 43):
            cv.drawMarker(img, (x, y), (158, 158, 158), cv.MARKER_TRIANGLE_UP, 10, 5)
        if (i > 27 and i < 32) or (i > 46 and i < 49):
            cv.circle(img, (x + 2, y + 2), 1, (158, 158, 158), 5)
        if (i > 31 and i < 41) or (i > 42 and i < 47) or (i < 28):
            cv.rectangle(img, (x, y), (x + 8, y + 8), (158, 158, 158), thickness=(cv.FILLED))
        i += 1

def draw_portrait(img, shape, sp=0, ep=17):
    pp = []
    for i in range(sp, ep):
        x, y = shape[i]
        point = [x, y]
        pp.append(point)

    for j in range(26, 17, -1):
        x1, y1 = shape[j]
        point1 = [x1, y1]
        pp.append(point1)

    x2, y2 = shape[0]
    point1 = [x2, y2]
    pp.append(point1)
    pp = np.array(pp, dtype=np.int32)
    cv.polylines(img, [pp], True, (220, 220, 0), thickness=4, lineType=cv.LINE_8)
    cv.fillPoly(img, [pp], (0, 0, 255, 0.5))

def draw_eye(img, shape, sp=36, ep=42, outc=(0, 0, 255), inc=(255, 255, 0)):
    points = []
    for i in range(sp, ep):
        x, y = shape[i]
        point = [x, y]
        points.append(point)
    points = np.array(points, dtype=np.int32)
    cv.polylines(img, [points], True, inc, thickness=4, lineType=cv.LINE_8)
    cv.fillPoly(img, [points], outc)

def draw_eyebrow(img, shape, sp=18, ep=22, lc=(0, 255, 0), fc=(0, 0, 255), sbp=36, bs=39, be=36):
    points = []
    x1, y1 = shape[sbp]
    point1 = [x1, y1]
    points.append(point1)
    for i in range(sp, ep):
        x, y = shape[i]
        point = [x, y]
        points.append(point)

    for j in range(bs, be, -1):
        x2, y2 = shape[j]
        point = [x2, y2]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    cv.polylines(img, [points], True, lc, thickness=4, lineType=cv.LINE_8)
    cv.fillPoly(img, [points], fc)

def draw_nose(img, shape, sp=27, ep=31):
    points = []
    points2 = []
    for i in range(sp, ep):
        x, y = shape[i]
        point = [x, y]
        points.append(point)

    for j in range(35, 29, -1):
        x2, y2 = shape[j]
        point2 = [x2, y2]
        points2.append(point2)

    points = np.array(points, dtype=np.int32)
    points2 = np.array(points2, dtype=np.int32)
    cv.polylines(img, [points], False, (220, 220, 0), thickness=4, lineType=cv.LINE_8)
    cv.polylines(img, [points2], True, (0, 255, 0), thickness=4, lineType=cv.LINE_8)
    cv.fillPoly(img, [points2], (255, 0, 0))

def draw_lip(img, shape, sp=48, ep=55, sp1=64, ep1=60, inc=(0, 255, 0), outc=(0, 255, 0)):
    points = []
    for i in range(sp, ep):
        x, y = shape[i]
        point = [x, y]
        points.append(point)

    for i in range(sp1, ep1, -1):
        x, y = shape[i]
        point = [x, y]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    cv.polylines(img, [points], True, outc, thickness=4, lineType=cv.LINE_8)
    cv.fillPoly(img, [points], inc)

faceDetector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
mona = cv.imread("mona liza.jpg")
monaGray = cv.cvtColor(mona, cv.COLOR_BGR2GRAY)
face = faceDetector.detectMultiScale(mona)
detector = dl.get_frontal_face_detector()
predictor = dl.shape_predictor("shape_predictor_68_face_landmarks.dat")
for (x, y, w, h) in face:
    sl = cv.rectangle(mona, (x, y), (x+w, y+h), (255, 0, 255), 2)
    cv.putText(mona, "Face : ", (x-10, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

rects = dl.rectangles()
for (x, y, w, h) in face:
    rects.append(dl.rectangle(x, y, x+w, y+h))
for (i, rect) in enumerate(rects):
    shape = predictor(mona, rect) #Gray mona
    shape = face_utils.shape_to_np(shape)
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv.rectangle(mona, (x, y), (x + w, y + h), (255, 0, 255), 2)
    cv.putText(mona, "Face #{}".format(i + 1), (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 40, 255), 2)
    for (x, y) in shape:
        cv.circle(mona, (x, y), 1, (0, 0, 255), -1)
    draw_portrait(mona, shape)
    draw_eyebrow(mona, shape)
    draw_eye(mona, shape) #left
    draw_eyebrow(mona, shape, 23, 27, (0, 220, 220), (220, 220, 0), 42, 45, 42)  # right brow
    draw_eye(mona, shape, 42, 48, (220, 220, 0), (0, 255, 0))  #right eye
    draw_nose(mona, shape)
    draw_lip(mona, shape)
    draw_lip(mona, shape, 54, 61, 67, 64, (0, 255, 255), (255, 0, 255)) #lover lips
    points(mona, shape)
cv.imshow("Lice sa maskom", mona)
#cv.imwrite("output.jpg", mona)
cv.waitKey(0)
cv.destroyAllWindows()
