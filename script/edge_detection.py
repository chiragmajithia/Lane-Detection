import sys,getopt
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import glob

def lineSegment(img,edges):
	#Detect lines in the image
	minLineLength = 10
	maxLineGap = 10
	lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
	for x in range(0, len(lines)):
		for x1,y1,x2,y2 in lines[x]:
			cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


	#Show image
	print lines.shape
	cv2.imshow("LSD",img )
	cv2.waitKey(0)

def hls_threshold(img,threshold=[100,100,100]):
	hls_img = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
	plt.subplot(411)
	plt.imshow(hls_img[:,:,0])
	plt.subplot(412)
	plt.imshow(hls_img[:,:,1])
	plt.subplot(413)
	plt.imshow(hls_img[:,:,2])
	mask = np.zeros_like(hls_img[:,:,1])
	mask[(hls_img[:,:,1]>100) & (hls_img[:,:,2]>60)] = 255;
	plt.subplot(414)
	plt.imshow(mask)
	plt.show()
	return mask

def canny(img,threshold=100):
	mask = cv2.Canny(img,180,200)
	return mask

def main():
	print "hello"
	img = cv2.imread('../test_images/straight_lines1.jpg')
	cv2.imshow("image",img)
	cv2.waitKey(1000)
	edges = canny(img)
	cv2.imshow("image",edges)
	cv2.waitKey(1000)
	mask = hls_threshold(img)
	kernel = np.ones((15,15), np.uint8)
	img_dilation = cv2.dilate(mask, kernel, iterations=1)
	res = cv2.bitwise_and(edges,edges,mask = mask)
	cv2.imshow("image",res)
	cv2.waitKey(10000)
	# lineSegment(img,edges)

if __name__ == '__main__':
	main()