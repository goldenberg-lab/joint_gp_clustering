import numpy as np
from numpy.linalg import inv, det

class GPCluster(object):
    """GP Class to compute likelihood"""
    def __init__(self, kernel, Y, T,var1=1, l=1,var2=1):
        """Initialization"""
        self.var1 = var1
        self.var2 = var2
        self.l = l
        self.Y = Y.reshape((-1,1)) # N.t x 1
        self.T = T # t x N
        self.kern = kernel
        self.cov = self.K(T) # N.t x N.t
        self.inv_cov = inv(self.cov)
        return

    def K(self,x):
        """
        Inputs
        x: t x N
        Returns
        K: N.t x N.t Covariance matrix
        """
        x = x.reshape((x.shape[0],-1))
        t = x.shape[0]
        Nt = x.shape[1]*t
        K = np.zeros((Nt, Nt))
        for i in range(Nt):
            for j in range(Nt):
                n1 = int(i/t)
                n2 = int(j/t)
                if n1 == n2:
                    K[i,j] = self.kern(x[i%t,n1],x[j%t,n2], self.l,self.var1) +\
                             self.kern(x[i%t,n1],x[j%t,n2], self.l,self.var2)
                else:
                    K[i,j] = self.kern(x[i%t,n1],x[j%t,n2], self.l,self.var1)
        return K

    def K_star(self,x,x_):
        """
        K_new = [[K , K*  ]
                [ K*, K**]]
        This function returns K*
        """
        t_ = x_.shape[0]
        t = x.shape[0]
        Nt = x.shape[1]*t
        K_star = np.zeros((t_, Nt))
        for i in range(t_):
            for j in range(Nt):
                n = int(j/t)
                K_star[i,j] = self.kern(x_[i],x[j%t,n], self.l,self.var1)
        return K_star

    def likelihood(self,y,t):
        """
        return the likelihood of a trajectory with data:
        y,t
        """
        y = y.reshape(-1,1)
        t = t.reshape(-1,1)
        K_star = self.K_star(self.T, t)
        K_star_star = self.K(t)
        temp = np.dot(K_star,self.inv_cov)
        mean = np.dot(temp,self.Y)
        inv_cov = inv(K_star_star - np.dot(temp, K_star.T))
        res_sqrt_abs_det_inv_cov = 1./np.sqrt(abs(det(inv_cov)))
        l = res_sqrt_abs_det_inv_cov * np.exp(-0.5 * np.dot(np.dot((y.T-mean.T),\
            inv_cov),(y-mean)))
        return l[0,0]

    def predict(self, t):
        """
        returns the mean and variance of the gp at time t
        """
        K_star = self.K_star(self.T, t)
        K_star_star = self.K(t)
        temp = np.dot(K_star,self.inv_cov)
        mean = np.dot(temp,self.Y)
        var = K_star_star - np.dot(temp, K_star.T)
        return mean, var

    def predict_y(self,y, t):
        """
        returns mean and variance
        """
        new_t = np.concatenate((self.T,t.reshape(-1,1)),axis=1)
        # print new_t.shape # TODO: check this
        new_y = np.concatenate((self.Y,y.reshape(-1,1)),axis=0) # TODO: check this
        # print new_y.shape
        K_ = self.K(new_t)
        K_star_star = K_[-1,-1]
        K_star = K_[-1,:-1] # TODO: check this
        K = K_[:-1,:-1] #TODO: check this
        inv_K = inv(K)
        temp = np.dot(K_star,inv_K)

        mean = np.dot(temp,new_y)
        var = K_star_star - np.dot(temp, K_star.T)
        return mean, var

    def likelihood_empty(self, y, t):
        """
        returns the likelihood of Y from the prior GP
        """
        K_star_star = self.K(t)
        inv_cov = inv(K_star_star)
        res_sqrt_abs_det_inv_cov = 1./np.sqrt(abs(det(inv_cov)))
        l = res_sqrt_abs_det_inv_cov * np.exp(-0.5 * np.dot(np.dot(y.T, inv_cov),y))
        return l
