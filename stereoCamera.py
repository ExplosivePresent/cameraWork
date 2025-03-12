import os
import numpy as np
import cv2 as cv

#Function to create point cloud file
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')


# calibration = np.load('C:/Users/dodod/PycharmProjects/cameraWork/calibrationData.npz', allow_pickle=False)
#
# leftMapX = calibration["leftMapX"]
# leftMapY = calibration["leftMapY"]
# leftROI = tuple(calibration["leftROI"])
# rightMapX = calibration["rightMapX"]
# rightMapY = calibration["rightMapY"]
# rightROI = tuple(calibration["rightROI"])

focal_length = 1/((1 / 2) + (1 / 40))
cap = cv.VideoCapture(0)    ###Change this to target stereo camera

# create and tune stereo object. Parameters from stereoTesting.py experimentation
#REF: https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
#		& openCV stereoBM documentation
numDisparities = 9 * 16	#can put all into containers, but these specifically for later normalizing/postprocessing
minDisparity = 15
stereo = cv.StereoBM.create(numDisparities = numDisparities, blockSize = 4*2+5)
#stereo.setNumDisparities(numDisparities)
#stereo.setBlockSize(3 *2 + 5)
stereo.setPreFilterType(1)
stereo.setPreFilterSize(3 *2 + 5)
stereo.setPreFilterCap(30)
stereo.setTextureThreshold(9)
stereo.setUniquenessRatio(27)
stereo.setSpeckleRange(10)
stereo.setSpeckleWindowSize(15 * 2)
stereo.setDisp12MaxDiff(13)
stereo.setMinDisparity(minDisparity)

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
	left = gray[:, :half]		#shape is (240,320)
	right = gray[:, half:]

	# #upsampling
	for i in range(1):
		h, w = left.shape
		left = cv.pyrUp(left, dstsize=(w * 2, h * 2))
		h, w = right.shape
		right = cv.pyrUp(right, dstsize=(w * 2, h * 2))

	disparity = stereo.compute(left, right)

###Additional postprocessing
	disparity = disparity.astype(np.float32)
	output = (disparity / 16.0 - minDisparity) / numDisparities	# Scaling down the disparity values and normalizing them
	#downsampling
	for i in range(2):
		h, w = left.shape
		left = cv.pyrDown(left, dstsize= (w//2, h // 2))
		h, w = right.shape
		right = cv.pyrDown(right, dstsize= (w//2, h // 2))
		h, w = disparity.shape
		disparity = cv.pyrDown(disparity, dstsize= (w//2, h // 2))

#####Point Cloud generation
	#REF: Parag-IIT point cloud from stereo, https://github.com/Parag-IIT/PointCloud-Generation-from-Stereo-imageso/blob/main/StereoSGBM.py
	Q = np.float32([[1, 0, 0, 0],
					 [0, -1, 0, 0],
					 [0, 0, focal_length * 0.05, 0],  # Focal length multiplication obtained experimentally.
					 [0, 0, 0, 1]])

	# Reproject points into 3D
	points_3D = cv.reprojectImageTo3D(disparity, Q)
	# Get color points
	colors = cv.cvtColor(left, cv.COLOR_BGR2RGB)
	# Generate point cloud
	print("\n Creating the output file... \n")
	create_output(points_3D, colors, 'pointCloud.pcd')


	# cv.namedWindow('stereo', cv.WINDOW_NORMAL)
	# cv.namedWindow('normal', cv.WINDOW_NORMAL)
	# cv.resizeWindow('stereo', (640, 480))
	# cv.resizeWindow('normal', (640, 480))
	# cv.imshow('normal', cv.addWeighted(left, .5, right, .5, 0))
	# cv.imshow('stereo', output)

	# if cv.waitKey(1) == ord('q'):
	# 	break

cap.release()
#cv.destroyAllWindows()
