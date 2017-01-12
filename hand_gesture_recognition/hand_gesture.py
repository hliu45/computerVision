# import the necessary packages
from pyimagesearch import imutils
import math
import numpy as np
import argparse
import cv2
 

class handProject(object):
	def __init__(self, skinLower, skinUpper):
		self.skinLower = skinLower
		self.skinUpper = skinUpper

	# extract image we want
	def extraction(self,skinImage, upper, lower):
		
		converted = cv2.cvtColor(skinImage, cv2.COLOR_BGR2HSV)

		# extract skin color from background
		skinMask = cv2.inRange(converted, upper, lower)

		# apply a series of erosions and dilations to the mask
		# using an elliptical kernel
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		skinMask = cv2.erode(skinMask, kernel, iterations = 1)
		skinMask = cv2.dilate(skinMask, kernel, iterations = 1)

		# blur the mask to help remove noise, then apply the
		# mask to the frame
		skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
		skinPicture = cv2.bitwise_and(skinImage, skinImage, mask = skinMask)


		return skinPicture

	def run(self):	

		# if a video path was not supplied, grab the reference
		# to the gray
		if not args.get("video", False):
			camera = cv2.VideoCapture(0)
 
		# otherwise, load the video
		else:
			camera = cv2.VideoCapture(args["video"])

		# hand training data
		#hand_cascade = cv2.CascadeClassifier('hand_1.xml')


		# keep looping over the frames in the video
		while (camera.isOpened()):
			# grab the current frame
			(grabbed, frame) = camera.read()
			frame = cv2.flip(frame,1)
 
			# if we are viewing a video and we did not grab a
			# frame, then we have reached the end of the video
			if args.get("video") and not grabbed:
				break


			# resize the frame, convert it to the HSV color space,
			# and determine the HSV pixel intensities that fall into
			frame = imutils.resize(frame, width = 900)

			# define our roi region
			x1, x2 = 10, 210
			y1, y2 = 600, 800
			roi_img = frame[x1:x2,y1:y2]

			# extract skin color from the background
			skin = self.extraction(roi_img, self.skinLower, self.skinUpper) 


			frameGray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
			skinGray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

	
			#  hand harrcascade part ----------------------------
			'''
			hand = hand_cascade.detectMultiScale(skinGray, 1.3, 5)
			for (x,y,w,h) in hand:
				cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
			'''
    		# ---------------------------------------------------------
    
			ret, skinGray = cv2.threshold(skinGray, 120, 255,cv2.THRESH_BINARY_INV)

			# find contour 
			image, contours, hierarchy = cv2.findContours(skinGray.copy(), \
				cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

			# find the dominant hand contour
			max_area = -1
			for i in range(len(contours)):
				cnt=contours[i]
				area = cv2.contourArea(cnt)
				if(area>max_area):
					max_area=area
					ci=i
			cnt=contours[ci]

			# draw contonur
			hull = cv2.convexHull(cnt)
			drawing = np.zeros(roi_img.shape,np.uint8)
			cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
			cv2.drawContours(drawing,[hull],0,(0,0,255),2)
			hull = cv2.convexHull(cnt,returnPoints = False)
			
			# check if defect angle smaller than 90 degree
			defects = cv2.convexityDefects(cnt,hull)
			count_defects = 0

			
			if defects != None:
				for i in range(defects.shape[0]):
					s,e,f,d = defects[i,0]
					start = tuple(cnt[s][0])
					end = tuple(cnt[e][0])
					far = tuple(cnt[f][0])
					a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
					b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
					c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
					angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
					if angle <= 90:
						count_defects += 1
						cv2.circle(roi_img,far,1,[0,0,255],3)
        	

        			cv2.line(roi_img,start,end,[0,255,0],3)
			print count_defects

        	

			# draw the ROI region
			cv2.rectangle(frame,(y2,x2),(y1,x1),(0,255,0),3)
 
			# show the skin in the image along with the mask
			cv2.imshow("images", np.hstack([roi_img, skin]))
			# show camera image
			cv2.imshow('camera', frame)
			#cv2.imshow('gray', skinGray)
			cv2.imshow('drawing',drawing)
 
			# if the 'q' key is pressed, stop the loop
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
 
		# cleanup the camera and close any open windows
		camera.release()
		cv2.destroyAllWindows()

if __name__ == '__main__':

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video",
		help = "path to the (optional) video file")
	args = vars(ap.parse_args())

	# define the upper and lower boundaries of the HSV pixel
	# intensities to be considered 'skin'
	skinlower = np.array([0, 30, 150], dtype = "uint8")
	skinupper = np.array([120, 255, 255], dtype = "uint8")

	
	handproject = handProject(skinlower,skinupper)
	handproject.run()
