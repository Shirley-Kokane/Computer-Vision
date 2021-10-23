import numpy as np
import numpy.matlib
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    h1, w1 = It.shape
    h2, w2 = It1.shape

    x1 = rect[0]
    x2 = rect[2]
    y1 = rect[1]
    y2 = rect[3]

    width = int(x2-x1)
    height = int(y2-y1)

    s1 = RectBivariateSpline(np.linspace(0, h1, num = h1, endpoint = False), np.linspace(0, w1, num= w1, endpoint = False), It)
    s2 = RectBivariateSpline(np.linspace(0,h2, num = h2, endpoint = False), np.linspace(0 , w2, num = w2, endpoint = False), It1)

    count = 1
    p = p0
    change = 1
    x,y = np.mgrid[x1:x2+1: width*1j, y1:y2+1:height*1j]
    while(change>threshold and count < 50):
        Itp = s1.ev(y, x).flatten()
        dxp = s2.ev(y+p[1], x+p[0],dy = 1).flatten()
        dyp = s2.ev(y+p[1], x+p[0],dx = 1).flatten()
        It1p = s2.ev(y+p[1], x+p[0]).flatten()
        
        A = np.zeros((width*height,2*width*height))
        for i in range(width*height):
            A[i,2*i] = dxp[i]
            A[i,2*i+1] = dyp[i]
        A = np.matmul(A,(numpy.matlib.repmat(np.eye(2),width*height,1)))
        
        b = np.reshape(Itp - It1p,(width*height,1))
        deltap = np.linalg.pinv(A).dot(b)
        change = np.linalg.norm(deltap)
        p = (p + deltap.T).ravel()
        count+=1
    
    return p
