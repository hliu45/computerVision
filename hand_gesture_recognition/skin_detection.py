# import the necessary packages
from pyimagesearch import imutils
import numpy as np
import argparse
import cv2
 

class handProject(object):
	def __init__(self, skinLower, skinUpper):
		self.skinLower = skinLower
		self.skinUpper = skinUpper

	# skin detection function
	def skindetection(self,skinImage):
		
		converted = cv2.cvtColor(skinImage, cv2.COLOR_BGR2HSV)

		# extract skin color from background
		skinMask = cv2.inRange(converted, self.skinLower, self.skinUpper)

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
		hand_cascade = cv2.CascadeClassifier('hand_1.xml')


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
			frame = imutils.resize(frame, width = 600)

			skin = self.skindetection(frame) 


			frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			skinGray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

	
			#  hand harrcascade part ----------------------------
			hand = hand_cascade.detectMultiScale(skinGray, 1.3, 5)
			for (x,y,w,h) in hand:
				cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
    		# ---------------------------------------------------------
    
			ret, skinGray = cv2.threshold(skinGray, 120, 255,cv2.THRESH_BINARY_INV)
 
			# show the skin in the image along with the mask
			cv2.imshow("images", np.hstack([frame, skin]))
			cv2.imshow('gray', skinGray)
 
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
