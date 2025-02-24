import numpy as np
import cv2 as cv
import os

vidPath = str('C:/Users/dodod/PycharmProjects/cameraWork/stereoVid.mp4')

cap = cv.VideoCapture(1)    ###Change this to target stereo camera

while True:
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #Take the resolution/shape dimensions and find half width
    h, w = gray.shape
    half = w // 2

    #split into left and right halves
    left = gray[:, :half]
    right = gray[:, half:]

    #Edge detector
    #    leftEdge = cv.Canny(left, 25, 250)
    #    rightEdge = cv.Canny(right, 25, 250)

####Downsample then upsample to work with less pixels and noise
    #downsampling
    for i in range(1):
        h, w = left.shape
        left = cv.pyrDown(left, dstsize= (w//2, h // 2))
        h, w = right.shape
        right = cv.pyrDown(right, dstsize= (w//2, h // 2))

    # #upsampling
    for i in range(1):
        h, w = left.shape
        left = cv.pyrUp(left, dstsize=(w * 2, h * 2))
        h, w = right.shape
        right = cv.pyrUp(right, dstsize=(w * 2, h * 2))


    #create stereo object
    stereo = cv.StereoBM.create(numDisparities=128, blockSize=21)
    #stereo.setTextureThreshold(5)
    stereo.setMinDisparity(4)
    stereo.setSpeckleRange(32)
    stereo.setSpeckleWindowSize(45)

    disparity = stereo.compute(left, right)

###Additional postprocessing
    depth_visualization_scale = 128  # used to shift image color from all grey to darker

    output = disparity / depth_visualization_scale
#    kernel = np.ones((3, 3), np.uint8)
#   kernel[1, 1] = 0
#    output = cv.erode(output, kernel, iterations=1)


    cv.namedWindow('stereo', cv.WINDOW_NORMAL)
    cv.namedWindow('normal', cv.WINDOW_NORMAL)
    cv.resizeWindow('stereo', (640, 480))
    cv.resizeWindow('normal', (640, 480))
    cv.imshow('normal', left)
    cv.imshow('stereo', output)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
