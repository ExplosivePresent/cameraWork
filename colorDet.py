import cv2
import numpy as np

#open camera feed with first camera in list
cam = cv2.VideoCapture(0)

while(1):
    #continually capture imgFrame from camera
    ret, imgFrame = cam.read()

    #convert RGB color space to hue-saturation-value
    hsvFrame = cv2.cvtColor(imgFrame, cv2.COLOR_BGR2HSV)

    #set up red mask to compare to
    red_lower = np.array([150, 87, 150], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    #dilate image for noise control
    kernel = np.ones((5, 5), "uint8")
    red_mask = cv2.dilate(red_mask, kernel)   #########################
    res_red = cv2.bitwise_and(imgFrame, imgFrame, mask=red_mask)

    #create contour coords for bounding box
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #bounding box
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imgFrame = cv2.rectangle(imgFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(imgFrame, "Red Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255))

    cv2.imshow("Red Color Detection in Real-Time", imgFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cam.release()
        cv2.destroyAllWindows()
        break