#!/usr/bin/python
import sys, getopt
import  numpy as np
import cv2
from matplotlib import pyplot as plt

global in_path,out_path,nx,ny,sqrEdgLen

def help():
	print "Input Arguments:\n1: Path to Calibration Images\n2: Path to Output Params\n3: Number of sqaures in X direction\n \
	4: Number of sqaures in Y direction\n5: Length of Edge of each square"

def dispParams():
	print "in_path: " + in_path
	print "out_path: "+ out_path
	print "nx: " + str(nx)
	print "ny: " + str(ny)
	print "sqrEdgLen: " + str(sqrEdgLen)  

def init():
	global nx,ny,objpnts,imgpnts,sqrEdgLen
	# Declare object points in 3d coordinates 
	# Z = 0 for all the points as reference frame is on the checker board
	objpnts = np.zeros((nx*ny,3),dtype=np.float32)
	objpnts[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)*sqrEdgLen

def main(argv):
	global in_path,out_path,nx,ny,sqrEdgLen
	if(np.shape(argv)[0] < 2):
		print "Error - Input Arguments Not Valid: Found " + str(np.shape(argv)[0]) + " Required 5"
		help()
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

	dispParams()

if __name__ == '__main__':
	main(sys.argv[1:])