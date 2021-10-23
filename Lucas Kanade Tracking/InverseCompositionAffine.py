import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    h1,w1 = It.shape
    h2,w2 = It1.shape
    change = 1
    A = np.zeros((h1*w1, 6))
    p0 = np.zeros(6)
    p = p0
    s1 = RectBivariateSpline(np.linspace(0,h1, num= h1, endpoint = False), np.linspace(0,w1, num=w1, endpoint = False), It)
    s2 = RectBivariateSpline(np.linspace(0, h2, num= h2, endpoint = False), np.linspace(0, w2, num = w2, endpoint = False), It1)

    x,y = np.mgrid[0:w1 , 0:h1]
    flat_x = np.reshape(x, (h1*w1, 1))
    flat_y = np.reshape(y, (h1*w1, 1))
    mat = np.vstack((x.reshape((1, h1*w1)), y.reshape((1, h1*w1)), np.ones((1 , h1*w1))))

    d_x = s1.ev(flat_x, flat_y, dy =1).reshape((h1*w1,1))
    d_y = s1.ev(flat_x, flat_y, dx = 1).reshape((h1*w1,1))
    #It1p = s1.ev(yind, xind).flatten()
    Itp = s1.ev(flat_x, flat_y).flatten()

    d_x  = np.reshape(d_x, (h1*w1,1))
    d_y = np.reshape(d_y, (h1*w1, 1))
    A[:,0] = np.multiply(flat_x,d_x).reshape((h1*w1, ))
    A[:,1] = np.multiply(flat_y,d_x).reshape((h1*w1, ))
    A[:,2] = d_x.reshape((h1*w1, ))
    A[:,3] = np.multiply(flat_x,d_y).reshape((h1*w1, ))
    A[:,4] = np.multiply(flat_y, d_y).reshape((h1*w1, ))
    A[:,5] = d_y.reshape((h1*w1, ))

    count = 0
    compute = np.matmul(np.linalg.pinv(np.matmul(A.T,A)),A.T)
    while (change > threshold and count < num_iters):
        M = np.array([[1 + p[0], p[1], p[2]],
                      [p[3], 1 + p[4], p[5]],
                      [0, 0, 1]])
        coorp = np.matmul(M, mat)
        xp = coorp[0]
        yp = coorp[1]

        It1p = s2.ev(yp, xp).flatten()

        b = np.reshape(Itp - It1p, (len(xp), 1))
        dp = compute.dot(b)

        change = np.linalg.norm(dp)
        p = (p + dp.T).ravel()
        # print(p)
        count += 1
    M = np.array([[1 + p[0], p[1], p[2]],
                  [p[3], 1 + p[4], p[5]],
                  [0, 0, 1]])
    return M


    return M
