# USAGE
# python skindetector.py
# python skindetector.py --video video/skin_example.mov

# import the necessary packages
from pyimagesearch import imutils
import numpy as np
import cv2


# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 30, 150], dtype = "uint8")
upper = np.array([140, 255, 255], dtype = "uint8")


camera = cv2.VideoCapture(0)

start = False
count = 0

# keep looping over the frames in the video
while (camera.isOpened()):
	# grab the current frame
	(grabbed, frame) = camera.read()
	frame = cv2.flip(frame,1)

	x1, x2 = 10, 210
	y1, y2 = 600, 800
	roi_img = frame[x1:x2,y1:y2]
	cv2.rectangle(frame,(y2,x2),(y1,x1),(0,255,0),3)

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video

	# resize the frame, convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
	#roi_img = imutils.resize(roi_img, width = 100)
	converted = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(converted, lower, upper)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
	skinMask = cv2.erode(skinMask, kernel, iterations = 1)
	skinMask = cv2.dilate(skinMask, kernel, iterations = 1)

	# blur the mask to help remove noise, then apply the
	# mask to the frame
	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	skin = cv2.bitwise_and(roi_img, roi_img, mask = skinMask)

	skinGray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
	ret, skinGray = cv2.threshold(skinGray, 120, 255,cv2.THRESH_BINARY_INV)
	print skinGray.shape

	'''
	# show the skin in the image along with the mask
	#cv2.imshow("images", np.hstack([frame, skin]))
	if start==True and count<=100:
		if count == 0:
			img = np.zeros(skinGray.shape,np.uint8)
			count+=1
		else:
			img = np.hstack([img,skinGray])
			count+=1

	print count
	cv2.imshow('skin',skin)
	cv2.imshow('skingray',skinGray)
	cv2.imshow('img',frame)
	'''
	# if the 'q' key is pressed, stop the loop
	k = cv2.waitKey(1) & 0xFF 
	if k == 27 or k == ord('q') or count>100:
		break
	if k == ord('s'):
		start = True

# cleanup the camera and close any open windows
'''
if start == True:
	cv2.imwrite('one.png',img)
'''
camera.release()
cv2.destroyAllWindows()