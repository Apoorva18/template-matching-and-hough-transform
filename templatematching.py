
import numpy as np

import cv2
# taken from the imutils package for python
def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized
 
# Performing the gaussian blur on the image
img= cv2.imread('pos_1.jpg')
blur = cv2.GaussianBlur(img,(3,3),0)
image= cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)


template = cv2.imread("template.png")
templategray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#Performing the laplacian on image and template
image = np.asarray(image)
imagelap = cv2.Laplacian(image,cv2.CV_8U)
templatelap = cv2.Laplacian(templategray,cv2.CV_8U)

resultant = None
for i in np.linspace(0.1,0.6,60):
	small = resize(templatelap, width = int(templatelap.shape[1] * i))
	print(small.shape)
	result = cv2.matchTemplate(imagelap, small, cv2.TM_CCOEFF)
	minvalue, maxvalue, minloc, maxloc = cv2.minMaxLoc(result)
	print(maxvalue)
	if resultant is None or maxvalue > resultant[0]:
		resultant = (maxvalue, maxloc)
		
(maxvalue, maxloc )= resultant

(x, y) = (int(maxloc[0] ), int(maxloc[1] ))
(x2, y2)= (int((maxloc[0] + 30)), int((maxloc[1] + 30) ))
if(maxvalue >140000 or maxvalue <130000):
	cv2.rectangle(img, (x-10, y-10), (x2, y2), (0, 0, 255), 2)
cv2.imshow("Image", img)
cv2.waitKey(0)


