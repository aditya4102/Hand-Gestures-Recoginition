#Importing
import cv2
import numpy as np
import math


vid = cv2.VideoCapture(0)

while True:
    flag, imgFlip = vid.read()
    img = cv2.flip(imgFlip,cv2.COLOR_BGR2GRAY)

    #Get hand data from rectangle
    cv2.rectangle(img, (100,100), (300,300), (0,255,0), 0)
    imgCrop = img[100:300, 100:300]


    imgBlur = cv2.GaussianBlur(imgCrop, (3,3), 0)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)

    lower = np.array([2,0,0])
    upper = np.array([20,255,255])
    mask = cv2.inRange(imgHSV, lower, upper)

    #Kernel for morphological transformation
    kernel = np.ones((5,5))

    #Apply morphological transforamtions to filter out the background noise
    dilation = cv2.dilate(mask,kernel, iterations=1)
    erosion = cv2.erode(dilation,kernel, iterations=1)

    filtered_img = cv2.GaussianBlur(erosion, (3,3), 0)
    ret, imgBin = cv2.threshold(filtered_img, 127, 255, 0)

    #cv2.imshow("Threshould img", imgBin)

    #find contours
    contours, hierarchy = cv2.findContours(imgBin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        contour = max(contours, key = lambda x: cv2.contourArea(x))

        #creating bounding  rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(imgCrop, (x,y), (x+w,y+h), (0,0,255), 0)

        #finding convex hull
        con_hull = cv2.convexHull(contour)


        #find convexity defects
        con_hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, con_hull)
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14

            #if angle > 90 draw a circle at the far point
            if angle<=90:
                count_defects+=1
                cv2.circle(imgCrop, far, 2, [0,0,255], -1)

            cv2.line(imgCrop, start, end, [0,255,0], 2)

        #print number
        if count_defects == 0:
            cv2.putText(img, "ONE", (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 2)
        elif count_defects == 1:
            cv2.putText(img, "TWO", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        elif count_defects == 2:
            cv2.putText(img, "THREE", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        elif count_defects == 3:
            cv2.putText(img, "FOUR", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        elif count_defects == 4:
            cv2.putText(img, "FIVE", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        else:
            pass

    except:
        pass

    

    cv2.imshow("Gesture", img)
    # cv2.imshow("Contours", all_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()




