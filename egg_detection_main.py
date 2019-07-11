import numpy as np
import pprint
import sys
import datetime
import math
import cv2
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import countmodule as ctm



width = 0
height = 0
eggCount = 0
exitCounter = 0
OffsetRefLines = 50  # Adjust ths value according to your usage
ReferenceFrame = None
distance_tresh = 200
radius_min = 0
radius_max = 0
area_min = 0
area_max = 0




#sys.exit(app.exec_())

def reScaleFrame(frame, percent=75):
    width = int(frame.shape[1] * percent // 100)
    height = int(frame.shape[0] * percent // 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def CheckInTheArea(coordYContour, coordYEntranceLine, coordYExitLine):
    if ((coordYContour <= coordYEntranceLine) and (coordYContour >= coordYExitLine)):
        return 1
    else:
        return 0


def CheckEntranceLineCrossing(coordYContour, coordYEntranceLine):
    absDistance = abs(coordYContour - coordYEntranceLine)

    if ((coordYContour >= coordYEntranceLine) and (absDistance <= 3)):
        return 1
    else:
        return 0


def getDistance(coordYEgg1, coordYEgg2):
    dist = abs(coordYEgg1 - coordYEgg2)

    return dist

for i in dir(cv2):
    print i,getattr(cv2,i)

cap = cv2.VideoCapture('img/20180910_144521.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
7
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('outpy.avi', fourcc, 10, (frame_width/3, frame_height/3))

#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('rtsp://admin:9ejq28Ez@172.16.1.65:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1')

fgbg = cv2.BackgroundSubtractorMOG()  # for mask
cc =1
cv2.setMouseCallback("image", ctm.click_and_crop)
while True:

    (grabbed, frame40) = cap.read()

    # if cc == 100:
    #     cv2.imwrite("myegg.jpg",frame)
    #     break
    egg_index = 0
    if not grabbed:
        print('Egg count: ' + str(egg_index))
        print('\n End of the video file...')
        break

    # get Settings radius/area values
    radius_min,radius_max = 1,60
    area_min,area_max = 200,4000
    cc =cc+1

    if radius_min == '':
        radius_min = 0
    if radius_max == '':
        radius_max = 0

    if area_min == '':
        area_min = 0
    if area_max == '':
        area_max = 0

    # frame40 = reScaleFrame(frame, percent=40)



    fgmask = fgbg.apply(frame40)

    eheight, ewidth, layers = frame40.shape
    new_h = eheight / 3
    new_w = ewidth / 3
    frame40 = cv2.resize(frame40, (new_w, new_h))
    height = np.size(frame40, 0)
    width = np.size(frame40, 1)

    hsv = cv2.cvtColor(frame40, cv2.COLOR_BGR2HSV)
    th, bw = cv2.threshold(hsv[:, :, 2], 10, 250, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    dist = cv2.distanceTransform(morph, cv2.cv.CV_DIST_L2, 3)
    borderSize = 35
    distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize,
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    gap = 10
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (borderSize - gap) + 1, 2 * (borderSize - gap) + 1))
    kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap,
                                 cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    distTempl = cv2.distanceTransform(kernel2, cv2.cv.CV_DIST_L2, 3)
    nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
    mn, mx, _, _ = cv2.minMaxLoc(nxcor)
    th, peaks = cv2.threshold(nxcor, mx * 0.5, 255, cv2.THRESH_BINARY)
    peaks8u = cv2.convertScaleAbs(peaks)
    contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    peaks8u = cv2.convertScaleAbs(peaks)  # to use as mask

    # plot reference lines (entrance and exit lines)
    coordYEntranceLine = (height // 2) + OffsetRefLines
    coordYMiddleLine = (height // 2)
    coordYExitLine = (height // 2) - OffsetRefLines
    cv2.line(frame40, (0, coordYEntranceLine), (width, coordYEntranceLine), (255, 0, 0), 2)
    cv2.line(frame40, (0, coordYMiddleLine), (width, coordYMiddleLine), (0, 255, 0), 6)
    cv2.line(frame40, (0, coordYExitLine), (width, coordYExitLine), (255, 0, 0), 2)

    flag = False
    egg_list = []

    egg_index=ctm.proces(contours,frame40,egg_index,coordYEntranceLine,coordYExitLine)
    print "count ",egg_index

    # for i in range(len(contours)):
    #     if  cv2.contourArea(contours[i]) >230:
    #         x, y, w, h = cv2.boundingRect(contours[i])
    #         _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y + h, x:x + w], peaks8u[y:y + h, x:x + w])
    #         # cv2.circle(frame40, (int(mxloc[0] + x), int(mxloc[1] + y)), int(mx), (255, 0, 0), 2)
    #         cv2.rectangle(frame40, (x, y), (x + w, y + h), (0, 255, 255), 2)
    #         cv2.drawContours(frame40, contours, i, (0, 0, 255), 2)
    # frame40 = reScaleFrame(frame40, percent=40)
    cv2.imshow("image", frame40)
    out.write(frame40)
    vis = np.concatenate(( peaks8u, nxcor), axis=1)
    cv2.imshow("Ori Frame", vis)

    key = cv2.waitKey(1)

    if key == 27:
        break

# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()




