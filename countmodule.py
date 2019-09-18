import numpy as np
import cv2
from datetime import datetime
import random



#Constant to reduce frame size and increase performance
r_framesize = 3
#Contour size
contour_size=1000
cam = cv2.VideoCapture(0)

#Image container to mark traffic path
img = np.zeros((cam.get(4)/r_framesize,cam.get(3)/r_framesize,3), np.uint8)
img_part = []
#I need to skip about 80 frames before processing begins
firstFrame = 80
#Count frame
count_frame=0
#Proximity to near vehicle
vcontour_distance=20


# fgbg = cv2.createBackgroundSubtractorMOG2()
# cv2.ocl.setUseOpenCL(False)

# Vehicle object
class vehicle(object):
    def __init__(self, id):
        self.id = id
        self.colour = int(random.randrange(2, 40, 2) * id)
        self.xy = []
        self.contour = None
        self.event_time = 0
        self.carspeed = 0

    def setContour(self, cont):
        self.contour = cont

    def set_time(self):
        self.event_time = datetime.now()

    def get_time(self):
        return self.event_time

    def get_speed(self, current_time):
        return self.get_map(current_time)

    def get_colour(self):
        return self.colour

    def getContour(self):
        return self.contour

    # Find the smallest value in the list with index
    def min_v(self, listvalues):
        m, index = min((v, i) for i, v in enumerate(listvalues))
        return (m, index)

    # Get euclidian distances between succesive frames
    def subTractCord(self, (xo, yo), (xn, yn)):
        o = xo - xn
        n = yo - yn
        ans = abs(o) + abs(n)
        return ans

    def get_speed(self, current_time):
        time_change = current_time - self.get_time()
        # Initial speed
        speed = 0
        # Make sure the vehicle cordinates obtained is enough to determine car speed
        if len(self.getXY()) >15 and self.carspeed == 0:

            coord = self.getXY()
            # Obtain first and last y cornidates of object list
            y_s = coord[0][1]
            y_e = coord[-1][1]
            distance = y_e - y_s
            if distance != 0:
                # Motion equation
                speed = float(distance / time_change.total_seconds())
                self.carspeed = int(abs(speed) / r_framesize)
        return (abs(speed) / r_framesize)

    # This method finds the current object in another frame
    def getDistance(self, list_contour):
        x, y = self.getMoment()
        an_list = []
        for c in list_contour:
            M = cv2.moments(c)
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            an = self.subTractCord((x, y), (cx, cy))
            an_list.append(an)
        match, index = self.min_v(an_list)
        return match, index

    def getId(self):
        return self.id

    def setId(self, value):
        self.id = value

    def getXY(self):
        return self.xy

    # Method add successive object cordinate
    def addXY(self, coord):
        self.xy.append(list(coord))

    # Method get centre of the object
    def getMoment(self):
        M = cv2.moments(self.contour)
        xy = (cx, cy) = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        return xy



    def drawVehicle(self, aframe, current_time, imgs):
        timedelta = self.get_speed(current_time)
        cv2.putText(aframe, str(self.getId()) + ".", self.getMoment(),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 240), 2)
        pts = np.array(self.getXY())
        pts = pts.reshape((-1, 1, 2))
        # col = int(self.get_colour())
        # cv2.polylines(aframe, [pts], False, (col + 1, col, col / 4), 4)
        # cv2.polylines(imgs, [pts], False, (0, 255, 0), 1)
        x, y, w, h = cv2.boundingRect(self.contour)
        cv2.rectangle(aframe, (x, y), (x + w, y + h), (255, 0, 0), 2)

#Detection and vanishing point
def get_contour_moment(contour,yent,yex):
    status = False
    M = cv2.moments(contour)
    xy = (cx, cy) = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    if xy[1] in range(yex,yent)and xy[0] in range(10,540):
        status = True
    return status


# You may need to find location with a click
def click_and_crop(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        print refPt


cv2.namedWindow('image')
cv2.setMouseCallback("image", click_and_crop)

#Default vehicle list
vlist =None
#Vehicle  count
count = 0

def proces(contour,pframe,ct,yent,yexi):
    global  count,vlist,img_part,vcontour_distance
    #Filter the detected contours
    cnts = [cnt for cnt in contour if cv2.contourArea(cnt) > 200  and get_contour_moment(cnt,yent,yexi)]  # fine tune this for miminum size
    #Check if first time contour found
    if vlist==None and len(cnts)>0:
        vlist=[]
        #cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            count += 1
            p = vehicle(count)
            p.set_time()
            p.setContour(c)
            p.addXY(p.getMoment())
            vlist.append(p)
            p.drawVehicle(pframe,datetime.now(),img)
        #vlist.sort(key=operator.attrgetter('id'))
    else:
        #If not first time check if there is  contour and that there is additional object
        if len(cnts)>0 :
            #Persist frame
            if len(cnts)<= len(vlist):
                nlist = vlist
                for ve in nlist:
                    dis,index = ve.getDistance(cnts)
                    if dis < vcontour_distance:
                        ve.setContour(cnts[index])
                        ve.addXY(ve.getMoment())
                        ve.drawVehicle(pframe,datetime.now(),img)
                    else:
                        vlist.remove(ve)
            #New vehichle   -remove contour as treated
            if len(cnts)> len(vlist):
                cpcontour = cnts
                nlist = vlist
                for ve in nlist:
                    dis, index = ve.getDistance(cpcontour)
                    if dis < vcontour_distance:
                        ve.setContour(cnts[index])
                        ve.addXY(ve.getMoment())
                        ve.drawVehicle(pframe,datetime.now(),img)
                        try:
                            cnts.remove(cnts[index])#remove index of contour
                        except:
                            print "erro"
                    else:
                        vlist.remove(ve)
                #THe remaining contour
                for c in cnts:
                    count = count + 1
                    p = vehicle(count)
                    p.setContour(c)
                    p.set_time()
                    p.addXY(p.getMoment())
                    vlist.append(p)
                    p.drawVehicle(pframe,datetime.now(),img)
    return count


def filter1_mask(fg_mask):
    sgray = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(sgray, learningRate=0.01)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Fill any small holes
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=2)
    return dilation


# Loop
# while True:
#     (grabbed, frame) = cam.read()
#     # if the frame could not be grabbed, then we have reached the end of the video
#     if not grabbed:
#         break
#     count_frame += 1
#     w, h, c = frame.shape
#     frame = cv2.resize(frame, (int(h) / r_framesize, int(w) / r_framesize))
#     thresh = filter1_mask(frame)
#
#     cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if count_frame >firstFrame:
#         proces(cnts, frame)
#     # Make sure there is queue of array
#     if len(img_part) >3:
#         img_pts = np.array(img_part)
#         img_pts = img_pts.reshape((-1, 1, 2))
#         cv2.polylines(img, [img_pts], False, (0, 255, 255), 4)
#     cv2.imshow("Security Feed", frame)
#
#     cv2.imshow("image", img)
#     # out.write(frame)
#     key = cv2.waitKey(2)
#
#     if key == ord('q'):
#         break
#
# # Release everything if job is finished
# cam.release()
# cv2.destroyAllWindows()