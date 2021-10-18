import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches
from PIL import Image

def matchPics(I1, I2, opts):
	#I1, I2 : Images to match
	#opts: input opts
	ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
	sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
	

	#Convert Images to GrayScale
	I3 = skimage.color.rgb2gray(I1)
	I4 = skimage.color.rgb2gray(I2)
	
	I3 = np.array(I3)
	I4 = np.array(I4)
	
	#Detect Features in Both Images
	
	locs1 = corner_detection(I3, sigma)
	locs2 = corner_detection(I4,sigma)
	
	
	
	#Obtain descriptors for the computed feature locations
	
	desc1, locs1 = computeBrief(I3,locs1)
	desc2,locs2 = computeBrief(I4,locs2)
	#Match features using the descriptors
	
	matches = briefMatch(desc1, desc2, ratio)
	
	#plotMatches(I3, I4, matches, locs1, locs2)

	return matches, locs1, locs2
  
  
