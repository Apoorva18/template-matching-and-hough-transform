import cv2
import numpy as np
import matplotlib.pyplot as plt
#read image
np.set_printoptions(threshold=np.nan)
img = cv2.imread('hough.jpg',0)
k,l = (img.shape)
#sobel for horizontal direction
kernel = ([-1,0,1],[-2,0,2],[-1,0,1])
m = k+2
n = l+2

matrix = [[0 for j in range(n)] for i in range(m)]
result = [[0 for x in range(n)] for  w in range(m)] 
result2 = [[0 for x in range(n)] for  w in range(m)]
i,j=0,0
#padding of image
for i in range(k):
    for j in range(l):
        matrix[i+1][j+1] = img[i][j]
        
sum1,y= 0,0
for z in range(k):
    for y in range(l):
        sum1=0
        for i in range(3):
            for j in range(3):
                sum1 += matrix[z+i][y+j]*kernel[i][j]
                
        result[z][y] = sum1
#finding the absolute value        
for i in range(k):
    for j in range(l):
        if(result[i][j]<0):
            result[i][j] = result[i][j] * (-1)

maxim = 0
for i in range(k):
    for j in range(l):
        if (result[i][j]>maxim):
            maxim = result[i][j]
            

#elimination of 0's
for i in range(k):
    for j in range(l):
        result[i][j] = result[i][j]/maxim
        
        
        
n1 = np.asarray(result)
#print(n1)
#cv2.imshow('image1',n1)
#cv2.waitKey(0)
nv = n1 *255
#cv2.imshow('horizontal.jpeg',nv)
#cv2.waitKey(0)
#sobel operator for x direction
kernel2 = ([-1,-5,-1],[0,0,0],[1,5,1])

sum2,y= 0,0
for z in range(k):
    for y in range(l):
        sum2=0
        for i in range(3):
            for j in range(3):
                sum2 += matrix[z+i][y+j]*kernel2[i][j]
                
        result2[z][y] = sum2

minim = 255
for i in range(k):
    for j in range(l):
        if (result2[i][j]<minim):
            minim = result2[i][j]
            
for i in range(k):
    for j in range(l):
        if(result2[i][j]<0):
            result2[i][j] = result2[i][j] * (-1)

maxim = 0
for i in range(k):
    for j in range(l):
        if (result2[i][j]>maxim):
            maxim = result2[i][j]
            


for i in range(k):
    for j in range(l):
         result2[i][j] = result2[i][j]/maxim
         
         
         
         
n2 = np.asarray(result2)

#cv2.imshow('image2',n2)
#cv2.waitKey(0)
nh = n2 *255
#cv2.imshow('vertical.jpeg',nh)
#cv2.waitKey(0)
#showing the image along both x and y direction edges
n3 = (n1**2 + n2**2)**(1/2)
for i in range(k):
	for j in range(l):
		if (n3[i][j]>0.2):
			matrix[i][j] = 255
		else:
			matrix[i][j] = 0
			
matrix = np.asarray(matrix)	
new2 = matrix/255		
#cv2.imshow('pbel',new2)
#cv2.waitKey(0)


thetas = np.linspace(0,50,51)			
theta = np.deg2rad(thetas)
max_length = int((k**2 + l**2 ) **0.5)

p = np.linspace(0,max_length,max_length)
#p = np.linspace(0,max_length,164,dtype=np.int)

count = np.zeros((len(theta),max_length))
     
xinput = []
yinput = []
for i in range(k):
	for j in range(l):
		if (new2[i][j]>0):
			xinput.append(j)
			yinput.append(k-i)
			

for t in range(len(theta)):			
	for i in range(len(xinput)):
		s = int(xinput[i]*np.cos(theta[t]) + yinput[i]*np.sin(theta[t]))
		count[t,s] +=1
	
array0 = [0 for i in range(len(count[2]))]
for i in range(len(count[2])):
	array0[i] = count[2][i]
arr = np.array(count[2 ])
#print(array0)
final  = []
inc = 90
i = 30
#final2 = arr.argsort()[-12:][::-1]


while i < len(count[2])-200:
	maxim = max(arr[i:i+100])
	
	arg = np.argmax(arr[i:i+100])
	
	final.append(i+arg)
	i = i + 100


img = cv2.imread('hough.jpg',1)
from math import pi
a = np.cos((pi/2) -theta[2])
b = np.sin((pi/2)- theta[2])
image = cv2.imread("hough.jpg",0)
for i in final:
    x0 = a*i
    y0 = b*i
    x1 = int(x0 +1000*(-b))
    y1 = int(y0 +1000*(a))
    
    x2 = int(x0 -1000*(-b))
    y2 = int(y0 -1000*(a))
    cv2.line(img,(y1,k-x1),(y2,k-x2),(0,0,255),2)	
cv2.imwrite('red_line.jpg',img)
#cv2.waitKey(0)

img = cv2.imread('hough.jpg',1)
array0 = [0 for i in range(len(count[37]))]
for i in range(len(count[37])):
	array0[i] = count[37][i]
arr = np.array(count[37])
#print(array0)
final2 = []
inc = 90
i = 40


while i < len(count[37])-100:
	maxim = max(arr[i:i+80])
	
	arg = np.argmax(arr[i:i+80])
	
	final2.append(i+arg)
	i = i + 80



from math import pi
a = np.cos((pi/2) -theta[37])
b = np.sin((pi/2)- theta[37])
image = cv2.imread("hough.jpg",0)
for i in final2:
    x0 = a*i
    y0 = b*i
    x1 = int(x0 +1000*(-b))
    y1 = int(y0 +1000*(a))
    x2 = int(x0 -1000*(-b))
    y2 = int(y0 -1000*(a))
    cv2.line(img,(y1,k-x1),(y2,k-x2),(255,0,0),2)
cv2.imwrite('blue_line.jpg',img)

