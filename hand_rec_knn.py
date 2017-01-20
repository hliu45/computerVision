import numpy as np
import cv2
from matplotlib import pyplot as plt

imgO = cv2.imread('one.png')
imgT = cv2.imread('two.png')
imgTh = cv2.imread('three.png')


img = np.vstack([imgO,imgT])
img = np.vstack([img,imgTh])

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cells = [np.hsplit(row,100) for row in np.vsplit(gray,3)]

# Now we split the image to 5000 cells, each 20x20 size
# Make it into a Numpy array. It size will be (50,100,20,20)


x = np.array(cells)

# Now we prepare train_data and test_data.
train = x[:,:100].reshape(-1,40000).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(1,4)

train_labels = np.repeat(k,100)[:,np.newaxis]

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)


# Create Label for one, two, three fingers
one =np.array([1])
two = np.array([2])
three = np.array([3])
onelabel = one[:,np.newaxis]
twolabel = two[:,np.newaxis]
threelabel = three[:,np.newaxis]

lower = np.array([0, 30, 150], dtype = "uint8")
upper = np.array([140, 255, 255], dtype = "uint8")


camera = cv2.VideoCapture(0)
while (camera.isOpened()):
	# grab the current frame
	(grabbed, frame) = camera.read()
	frame = cv2.flip(frame,1)

	x1, x2 = 10, 210
	y1, y2 = 600, 800
	roi_img = frame[x1:x2,y1:y2]
	cv2.rectangle(frame,(y2,x2),(y1,x1),(0,255,0),3)

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

	test = skinGray.reshape(-1,40000).astype(np.float32)

	ret,result,neighbours,dist = knn.findNearest(test,k=5)

	if result==onelabel:
		cv2.putText(frame,"one finger..1", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(250,250,250), 2, 2)
	elif result==twolabel:
		cv2.putText(frame,"two fingers..2", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(250,250,250), 2, 2)
	elif result==threelabel:
		cv2.putText(frame,"three fingers..3", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(250,250,250), 2, 2)
	else:
		cv2.putText(frame,"...............", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(250,250,250), 2, 2)

	cv2.imshow('skingray',skinGray)
	cv2.imshow('img',frame)

	k = cv2.waitKey(1) & 0xFF 
	if k == 27 or k == ord('q'):
		break

camera.release()
cv2.destroyAllWindows()




