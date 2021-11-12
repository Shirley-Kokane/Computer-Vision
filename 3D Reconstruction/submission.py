"""
Homework4.
Replace 'pass' by your implementation.
"""

import numpy as np
import sympy
import scipy.optimize as spo
import os.path
from util import _singularize
from util import refineF
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = np.divide(pts1, M)
    pts2 = np.divide(pts2, M)

    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]

    l1 = x1*x2
    l2 = x1*y2
    l3 = x1
    l4 = y1*x2
    l5 = y1*y2
    l6 = y1
    l7 = x2
    l8 = y2
    l9 = np.ones(len(l8))

    A = np.vstack((l1,l2,l3, l4, l5, l6, l7, l8, l9))

    

    u, s,vh = np.linalg.svd(A.T)

    f_mat = vh[-1,:]

    F = f_mat.reshape(3,3)
    singF = _singularize(F)
    F = refineF(singF, pts1, pts2)
    T = np.array([[1./M,0,0],[0,1./M,0],[0,0,1]])
    F = np.matmul(T.T,np.matmul(F,T))

    np.savez('q2_1.npz', M=M, F=F)

    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    K2T = K2.T
    essential = np.matmul(K2T,np.matmul(F,K1))
    np.savez('q3_1.npz', E = essential, F=F)
    return essential


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    P = np.zeros((pts1.shape[0],3))
    total_p = np.zeros((pts1.shape[0], 4))

    for i in range(pts1.shape[0]):
        x1 = pts1[i,0]
        x2 = pts2[i,0]
        y1 = pts1[i,1]
        y2 = pts2[i,1]
        A1 = x1*C1[2,:] - C1[0,:] #[C1[0:1, 0:2], x1 - C1[0,3]]
        A2 = y1*C1[2,:] - C1[1,:] #[C1[1:2, 0:2], y1 - C1[1,3]]
        A3 = x2*C2[2,:] - C2[0,:] #[C2[0:1, 0:2], x2 - C2[0,3]]
        A4 = y2*C2[2,:] - C2[1,:] #[C2[1:2, 0:2], y2 - C2[1,3]]
        A = np.vstack((A1,A2,A3,A4))
        U1,U2, vh = np.linalg.svd(A)
        p = vh[-1, :]
        p = p/p[3]
        
        P[i,:] = [p[0], p[1], p[2]]
        total_p[i,:] = p
    trans_p = total_p.T
    print(C1.shape, trans_p.shape)
    p1_predict = np.matmul(C1, trans_p)
    p1_norm = p1_predict/p1_predict[-1,:]
    p2_predict = np.matmul(C2, trans_p)
    p2_norm = p2_predict/p2_predict[-1,:]
    p1_trans = p1_norm[[0,1],:].T
    p2_trans = p2_norm[[0,1],:].T
    err1 = np.sum((p1_trans - pts1)**2)
    err2 = np.sum((p2_trans - pts2)**2)
    err = err1 + err2

    return P, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    data = np.load('./data/some_corresp.npz')
    epline = F.dot(np.array([[x1], [y1], [1]]))
    a= epline[0]
    b = epline[1]
    c = epline[2]

    yline = np.arange(y1-30, y1+30)
    h,w,channel = im1.shape

    xline = (-(b*yline + c)/a)

    image1 = gaussian_filter(im1, output=np.float64, sigma=1)
    image2 = gaussian_filter(im2, output= np.float64, sigma =1)
    errmin = np.inf
    res = 0
    for i in range(60):
        x2 = int(xline[i])
        y2 = yline[i]
        # print(x2,y2)
        if (x2>=5  and x2<= w-5-1 and y2>=5 and y2<= h-5-1):
            patch1 = image1[y1-5:y1+5+1, x1-5:x1+5+1,:]
            patch2 = image2[y2-5:y2+5+1, x2-5:x2+5+1,:]
            diff = (patch1-patch2).flatten()
            err= (np.sum(diff**2))
            if (err<errmin):
                errmin = err
                res = i
    np.savez('q4_1.npz', F=F, pts1=data['pts1'], pts2=data['pts2'])

    return xline[res], yline[res]

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.3):
    # Replace pass by your implementation
    total_inlier = 0
    for i in range(0 ,200):
        index = np.random.choice(pts1.shape[0], 8)
        select_pts2 = pts2[index,:]
        select_pts1 = pts1[index, :]
        all_F = eightpoint(select_pts1, select_pts2, M)
        print(all_F.shape)
        
        F = all_F
        inliers_now = []
        for k in range(0, pts2.shape[0]):
            x1 = np.append(pts1[k,:],1).reshape((3,1))
            x2 = np.append(pts2[k, :], 1).reshape((1,3))
            error = abs(np.matmul(x2, np.matmul(F, x1)))
            if (error) < 0.002:
                inliers_now.append(k)
        if (len(inliers_now) > total_inlier):
            total_inlier = len(inliers_now)
            all_inliers = inliers_now
        inliers1 = pts1[all_inliers,:]

        inliers2 = pts2[all_inliers,:]

        F = eightpoint(inliers1, inliers2, M)

    return F, all_inliers

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    mat = np.linalg.norm(r)
    if(mat == 0):
        return np.eye(3)
    a = r/mat
    a3 = a[2,0]
    a2 = a[1,0]
    a1 = a[0,0]

    final_a = np.zeros((3,3))

    final_a[0,0] = 0
    final_a[0,1] = -a3
    final_a[0,2] = a2
    final_a[1,0] = a3
    final_a[1,1] = 0
    final_a[1, 2] = -a1
    final_a[2, 0] = -a2
    final_a[2,1] = a1
    final_a[2,2] = 0

    R = np.eye(3)*np.cos(mat)+(1-np.cos(mat))*a.dot(a.T)+np.sin(mat)*final_a 
    return R

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    pass

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation

    pass


