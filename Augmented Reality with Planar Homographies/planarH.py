import numpy as np
import cv2
from opts import get_opts
from matchPics import matchPics
from scipy.spatial import distance
from math import dist
from helper import *
from PIL import Image

opts = get_opts()


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points


	#img1 = cv2.imread('../data/cv_cover.jpg')
	#img2 = cv2.imread('../data/cv_desk.jpg')

	#matches, x1 , x2 = matchPics(x1, x2, opts)
	#print(x1.shape)
	#print(x2.shape)
	#print(matches.shape)
	#return H2to1
	coor,N = np.shape(x1)
	u = x1[:,0].reshape(x2.shape[0],1)
	v = x1[:,1].reshape(x2.shape[0],1)
	x = x2[:,0].reshape(x2.shape[0],1)
	y = x2[:,1].reshape(x2.shape[0],1)
	all_A = np.zeros((len(x1)*2 , 9))

	for i in range(0, len(x1)):
		t1, y1,u1,v1 = x1[i,0] , x1[i,1], x2[i,0], x2[i,1]
		#A1 = np.hstack((-1*u1,-1*v1,-1*np.ones((N,1)),np.zeros((N,3)),np.multiply(t1,u1),np.multiply(t1,v1),x))
		A1 = np.hstack(( [-u1, -v1, -1],np.zeros(3), [np.multiply(t1,u1), np.multiply(t1,v1), t1]))
		#A2 = np.hstack((np.zeros((N,3)),-1*u1,-1*v1,-1*np.ones((N,1)),np.multiply(y1,u1),np.multiply(y1,v1),y1))
		A2 = np.hstack((np.zeros((3)), [-u1, -v1, -1] ,  [np.multiply(u1,y1), np.multiply(y1,v1), y1]))

		#A = np.vstack((A1,A2))
		#print(A)
		all_A[2*i,:] = A1
		all_A[2*i+1, :] = A2
	#	all_A.append(A)
	#A_up = np.hstack((-1*u,-1*v,-1*np.ones((N,1)),np.zeros((N,3)),np.multiply(x,u),np.multiply(x,v),x))
	#A_down = np.hstack((np.zeros((N,3)),-1*u,-1*v,-1*np.ones((N,1)),np.multiply(y,u),np.multiply(y,v),y))
	#A = np.vstack((A_up,A_down))
	#print(np.isnan(all_A))
	w,v, vh = np.linalg.svd(all_A)
	#print(vh.shape)
	hto1 = vh[-1,:]

	final_h = hto1.reshape((3,3))
	final_h = final_h/final_h[2,2]

	return final_h



def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	t1 = [t[0] for t in x1]
	y1 = [t[1] for t in x1]
	t2 = [t[0] for t in x2]
	y2 = [t[1] for t in x2]

	cent1 = np.zeros((1,2))
	cent2 = np.zeros((1,2))
	cent1[0,0], cent1[0,1] = sum(t1)/len(t1), sum(y1)/len(t1)
	cent2[0,0], cent2[0,1] = sum(t2)/len(t2), sum(y2)/len(t2)

	#Shift the origin of the points to the centroid
	#print(cent1, cent2)
	x1_shift = np.zeros((x1.shape))
	x1_shift[:,0] = x1[:,0] - cent1[0,0]
	x1_shift[:,1] =  x1[:,1] - cent1[0,1]
	x2_shift = np.zeros((x2.shape))
	x2_shift[:,0] = x2[:,0] - cent2[0,0]
	x2_shift[:,1] = x2[:,1] - cent2[0,1]

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	new_cent1 = np.broadcast_to([0,0], x1.shape)
	#print(cent1.shape, x1.shape)
	all_dist_1 =  np.linalg.norm(x1_shift-new_cent1,axis=1)
	max_dist_1 = np.max(all_dist_1)
	x1_norm = np.sqrt(2)*x1_shift/max_dist_1

	#cent2 = np.broadcast_to(cent2, x2.shape)
	all_dist_2 = np.linalg.norm(x2_shift-new_cent1, axis=1)
	max_dist_2 = np.max(all_dist_2)
	x2_norm = np.sqrt(2)*x2_shift/max_dist_2
	#Similarity transform 1
	t1 = np.array([[np.sqrt(2)/max_dist_1, 0, -np.sqrt(2)*cent1[0,0]/max_dist_1], [0, np.sqrt(2)/max_dist_1, -np.sqrt(2)*cent1[0,1]/max_dist_1], [0, 0, 1]])
	#print(t1.shape)
	#t1 = np.int(t1)
	#x1_norm = t1*x2

	#Similarity transform 2
	t2 = np.array([[np.sqrt(2)/max_dist_2, 0, -np.sqrt(2)*cent2[0,0]/max_dist_2], [0, np.sqrt(2)/max_dist_2, -np.sqrt(2)*cent2[0,1]/max_dist_2], [0, 0, 1]])
	#x2_norm = t2*x2
	#Compute homography
	hto1 = computeH(x1_norm, x2_norm)

	#print(hto1)
	#Denormalization
	t1_inverse = np.linalg.inv(t1)
	mlt1 = np.matmul(t1_inverse, hto1)
	hto1_final = np.matmul(mlt1,t2)
	hto1_final = hto1_final/hto1_final[2][2]
	#print(x2)
	#print(np.matmul(hto1_final,x1))

	return hto1_final




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
	#matches, locs1,locs2 = matchPics(img1,img2,opts)
	matched_locs1 = np.vstack((locs1[:,1] , locs1[:,0])).T
	#print(matched_locs1.shape)
	matched_locs2 = np.vstack((locs2[:,1] , locs2[:,0])).T
	#matched_locs1 = locs1[matches[:,0]]
	#matched_locs1 = matched_locs1[:,[0,1]]
	#matched_locs2 = locs2[matches[:,1]]
	#matched_locs2 = matched_locs2[:, [0,1]]
	N = matched_locs1.shape[0]
	
	homo1 = np.vstack((matched_locs1.T, np.ones((1,N))))
	homo2 = np.vstack((matched_locs2.T, np.ones((1,N))))
	#print(homo1.shape)
	inlier_count = 0
	for i in range(0, max_iters):
		pairs = np.random.choice(N,4)
		#print(pairs)
		p1 = matched_locs1[pairs,:]
		p2 = matched_locs2[pairs,:]


		homography_mat = computeH_norm(p1,p2)
		#print(homography_mat)
		expected_homo1 = np.matmul(homography_mat, homo2)

		divide_homo1 = expected_homo1[-1,:]
		expected_homo1 = expected_homo1/divide_homo1
		#print(expected_homo1)
		sub = (expected_homo1 -  homo1)#[[0,1], :]
		diff = np.linalg.norm(sub, axis=0)
		inlier_present = np.where(diff< inlier_tol)

		#print(pairs,homo1.shape, sub.shape, p1.shape)
		#print(diff.shape)
		#print('Shape of inlier', inlier_present)
		#print('Done')
		if(np.shape(inlier_present)[1] > inlier_count):
			inlier_count = np.shape(inlier_present)[1]
			final_inlier = inlier_present 

	#print(homo1)
	#print(expected_homo1)
	#print('Inlier present', inlier_present)
	#print('Final inlier count ', inlier_count)

	t1 = matched_locs1[final_inlier]
	t2 = matched_locs2[final_inlier]

	finalh = computeH_norm(t1, t2)
	finalh = finalh/finalh[2,2]
	return finalh,t1



#print(h.shape)

#	return bestH2to1, inliers

#img1 = cv2.imread('../data/cv_cover.jpg')
#img2 = cv2.imread('../data/cv_cover.jpg')
#matches,locs1, locs2 = matchPics(img1, img2, opts)
#x1 = locs1[matches[:,0]]
#x2 = locs2[matches[:,1]]
#finalh = computeH_norm(x1,x2)

#finalh = computeH_ransac(x1, x2, opts)
#print(finalh)


def compositeH(H2to1, template, img):
	
	sigma = opts.sigma
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*img
	#For warping the template to the image, we need to invert it.
	
	
	#Create mask of same size as template
	mask1 = np.ones((template.shape[0], template.shape[1]))
	#Warp mask by appropriate homography
	mask_warp1 = cv2.warpPerspective(mask1, H2to1, (img.shape[1], img.shape[0]))
	mask_warp1 = mask_warp1.astype(np.int32)
	#print(mask_warp1)
	#Image.fromarray(mask_warp1).show()
	#print(mask_warp1.shape)
	#Warp template by appropriate homography
	template_warp1 = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))
	#print(np.max(template_warp1), np.min(template_warp1))
	#print(template_warp1.shape)
	#Image.fromarray(template_warp1, 'RGB').show()

	#Use mask to combine the warped template and the image
	final_img = img.copy()

	for i in range(0,img.shape[0]):

		for j in range(0,img.shape[1]):

			for k in range(0,img.shape[2]):


				if template_warp1[i,j,k]  != 0:
					#print(i,j)
					final_img[i,j,k] = template_warp1[i,j,k]
				#else:

				#	final_img[i,j,k] = img[i,j,k]
	#final_img = np.transpose(final_img, (2,0,1))

	return final_img
	


