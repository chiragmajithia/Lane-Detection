#!/usr/bin/python
import sys, getopt
import  numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

global in_path,out_path,nx,ny,sqrEdgLen

def parseArg(argv):
	global in_path,out_path,nx,ny,sqrEdgLen
	if(np.shape(argv)[0] < 2):
		print "Error - Input Arguments Not Valid: Found " + str(np.shape(argv)[0]) + " Required 5"
		help()
		exit()
	elif(np.shape(argv)[0] == 5):
		in_path = argv[0]
		out_path = argv[1]
		nx = int(argv[2])
		ny = int(argv[3])
		sqrEdgLen = float(argv[4])
	elif(np.shape(argv)[0] == 2):
		in_path = argv[0]
		out_path = argv[1]
		nx = 9
		ny = 6
		sqrEdgLen = 30.0

def help():
	print "Input Arguments:\n1: Path to Calibration Images\n2: Path to Output Params\n3: Number of sqaures in X direction\n \
	4: Number of sqaures in Y direction\n5: Length of Edge of each square"

def dispParams():
	print "in_path: " + in_path
	print "out_path: "+ out_path
	print "nx: " + str(nx)
	print "ny: " + str(ny)
	print "sqrEdgLen: " + str(sqrEdgLen)  

def testFindCheckBoardCorners(img_no=2):
	#Test function to generate corners from a checkboard image
	global nx,ny,sqrEdgLen
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	
	print "intializing object points.."
	# Declare object points in 3d coordinates 
	# Z = 0 for all the points as reference frame is on the checker board
	objpnts = np.zeros((nx*ny,3),dtype=np.float32)
	objpnts[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)*sqrEdgLen

	print "Loading image .. " + str(img_no)
	fname = in_path+'calibration{}.jpg'.format(img_no)
	img = cv2.imread(fname)
	if(img is None):
		print "Unable to read image from" + fname
		exit()

	#Fing chess board corners
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray,(nx,ny),cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FILTER_QUADS)
	if ret == True:
    # Draw and display the corners
		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
		plt.imshow(img)
		plt.title("Chessboard image with corners")
		plt.show()
	else:
	 	cv2.imshow("image",img);
	 	cv2.waitKey(0)

def getCorrespondences(vis=False):
	global nx,ny,sqrEdgLen
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	
	print "intializing object points.."
	# Declare object points in 3d coordinates 
	# Z = 0 for all the points as reference frame is on the checker board
	objpnts = np.zeros((nx*ny,3),dtype=np.float32)
	objpnts[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)*sqrEdgLen

	img_points = []
	obj_points = []

	print "Loading Images from " + in_path
	#Load Images
	n_imgs = len(glob.glob(in_path+"*.jpg"))

	for i in range(1,n_imgs):
		fname = in_path+'calibration{}.jpg'.format(i)
		img = cv2.imread(fname)
		#Check if image is read sucessfully
		if(img is None):
			print "Unable to read image from" + fname
			exit()

		#Compute corners
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray,(nx,ny),cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FILTER_QUADS)
		if ret == True:
	    # Draw and display the corners
			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			img_points.append(corners2)
			obj_points.append(objpnts)
			if vis:
				cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
				cv2.imshow("image",img)
			 	cv2.waitKey(10)
	return obj_points,img_points,img.shape[:2]

def getCameraMatrix():
	objpoints,imgpoints,img_size = getCorrespondences(False);
	err, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
	return mtx,dist

def undistort(img,mtx,dist,img_size):
	#generate undistorted version of a image img.
	nmtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,img_size,1,img_size)
	cv2.imshow('undistorted image',cv2.undistort(img,mtx,dist,None,nmtx))
	cv2.waitKey(0)

def main(argv):
	parseArg(argv)
	print "Parsed command line input:"
	dispParams()
	#testFindCheckBoardCorners(2)
	objpoints,imgpoints,img_size = getCorrespondences(False);
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
	fname = in_path+'calibration{}.jpg'.format(1)
	img = cv2.imread(fname)
	undistort(img,mtx,dist,img_size)

if __name__ == '__main__':
	main(sys.argv[1:])