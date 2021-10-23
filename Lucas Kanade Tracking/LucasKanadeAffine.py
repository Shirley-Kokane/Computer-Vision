import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    h1,w1 = It.shape
    h2,w2 = It1.shape

    p0 = np.zeros(6)
    s1= RectBivariateSpline(np.linspace(0, h1, num = h1, endpoint = False), np.linspace(0, w1, num= w1, endpoint = False), It)
    s2 = RectBivariateSpline(np.linspace(0,h2, num= h2, endpoint = False), np.linspace(0,w2,num=w2, endpoint = False), It1)

    x,y = np.mgrid[0:w1, 0:h1]
    flat_x = x.reshape((h1*w1,1))
    flat_y = y.reshape((h1*w1,1))
    mat = np.vstack((x.reshape((1, h1*w1)), y.reshape((1, h1*w1)), np.ones((1,h1*w1))))
    change = 1
    count = 0
    p =p0
    deriv_x , deriv_y = np.gradient(It)
    mask = np.ones((h1, w1)).astype(np.float32)
    A = np.zeros((w1*h1, 6)).astype(np.float32)
    b = np.zeros((w1*h1)).astype(np.float32)
	

    while change>=threshold and count< num_iters:
        #print(M.shape)
        mult = np.matmul(M, mat)
       
        
        x_new = (flat_x[flat_x <= w1])
        y_new = (flat_y[flat_y <= h1])
        

        x_new = x_new.reshape((len(x_new), 1))
        y_new = y_new.reshape((len(y_new),1))


        xind = mult[0][mult[0] <= w1]
        xind = xind.reshape((len(xind),1))
        yind = mult[1][mult[1] <= h1]
        yind = yind.reshape((len(yind),1))

        min_len = np.amin([len(x_new), len(y_new), len(xind), len(yind)])
        xind = xind[:min_len]
        yind = yind[:min_len]
        x_new = x_new[:min_len]
        y_new = y_new[:min_len]
        #print(xind.shape, yind.shape)
        d_x = s2.ev(yind, xind, dy =1).reshape((len(xind),1))
        d_y = s2.ev(yind, xind, dx = 1).reshape((len(xind),1))
        It1p = s2.ev(yind, xind).reshape((len(xind),1))
        Itp = s1.ev(y_new, x_new).reshape((len(x_new),1))

        A = np.zeros((len(x_new), 6)).astype(np.float32)
        A[:,0] = (x_new.reshape((len(x_new),1))*d_x).reshape((len(x_new),))
        A[:,1] = (x_new.reshape((len(x_new),1))*d_y).reshape((len(x_new),))
        A[:,2] = (y_new.reshape((len(x_new),1))*d_x).reshape((len(x_new),)) 
        A[:,3] = (y_new.reshape((len(x_new),1))*d_y).reshape((len(x_new),))
        A[:,4] = d_x.reshape((len(x_new),))
        A[:,5] = d_y.reshape((len(x_new),))
        #print(A.shape)
        b = np.reshape(Itp - It1p, (len(x_new), 1))
        
        #A = np.hstack([affine_derivx[:]*flat_x[:], affine_derivx[:]*flat_y[:], affine_derivx, affine_derivy[:]*flat_x[:],affine_derivy[:]*flat_y[:],  affine_derivy]).reshape(( mat.shape[1],6))

        #dp, residuals, rank, s = np.linalg.lstsq(A, b)
        
        dp = np.linalg.pinv(A).dot(b)
        change = np.linalg.norm(dp)
        p = (p+dp.T).ravel()
        M = np.array([[1 + p[0], p[2], p[4]], [p[1], 1+ p[3], p[5]]])
        count = count +1
    M = np.append(M, [[0,0,1]], axis=0)
    #M = np.linalg.inv(M)
    return M
    