import numpy as np
import scipy


def matker32(x1,x2,l=1, sig2 = 1):
    #dist = lambda p1, p2: np.linalg.norm(p1 - p2, axis=0)
    #r = np.asarray([[dist(p1,p2) for p2 in x1] for p1 in x2])
    r = scipy.spatial.distance.cdist(x1,x2).transpose()
    #given the distance r
    p1 = (np.sqrt(3)*r)/l
    p2 = np.exp(-p1)
    p3 = 1 + p1
    return(sig2 * p2 * p3) 


class cagp:
    def __init__(self):
        pass
    
    def CAGP_Post(self, Xtrain, ytrain, Xtest, l, sigma2, sig_ker2, m, ytest = None,  m_0tr = None, m_0ts = None, PLS = 'GS'):
        self.Xtest = Xtest
        self.ytest = ytest
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtrdim = Xtrain.shape[0]
        self.Xtsdim = Xtest.shape[0]

        #lengthscale of kernel
        self.l = l
        #amplitude of the kernel
        self.sig_ker2 = sig_ker2
        self.sigma2 = sigma2

        self.V = matker32(Xtrain, Xtest, l, self.sig_ker2).T 
        self.G = matker32( Xtrain, Xtrain, l, self.sig_ker2) + (sigma2*np.eye(self.Xtrdim))
        self.CXtestXest = matker32(self.Xtest, self.Xtest, self.l, self.sig_ker2)

        comp_vec = ytest
        if comp_vec is None:
            print('ytest not provided. RMSE and NLL values are not reliable.')
            self.actual_values = np.zeros(self.Xtsdim)
        else:
            self.actual_values = comp_vec
        
        self.m_0ts = m_0ts
        if m_0ts is None:
            self.m_0ts = np.zeros(self.Xtsdim)
        
        if m_0tr is None:
            self.m_0tr = np.zeros(self.Xtrdim)
        self.b = ytrain - self.m_0tr


        if PLS == 'GS':
            self.PLS_GS(m)
            m = self.m
            k = self.k

        if PLS == 'CG':
            self.PLS_CG(m)
            m = self.mcg
            k = self.C_cg

        return m, k


    #This gives the posterior of solution to the linear system Gv = b
    def PLS_GS(self, m):
        L = np.tril(self.G)
        U = np.triu(self.G, 1)
        D = np.diag(self.G)
        sqD = np.sqrt(D)[:,None]
        #change for V here?? - no
        Z = scipy.linalg.solve_triangular(L.T, self.V ,lower=False,check_finite=False)
        f = scipy.linalg.solve_triangular(L,self.b,lower=True,check_finite=False)
        M = sqD * Z
        self.VDmV = M.T @ M
        self.xm = np.zeros(self.Xtrdim)
        self.k =  self.CXtestXest - self.VDmV
        self.vm = self.V.T @ self.xm
        self.m = self.m_0ts + self.vm
        self.xm = np.zeros(self.Xtrdim)
        self.rmse = np.zeros(m+1)
        self.nll = np.zeros(m+1)
        self.rmse[0] = np.linalg.norm(self.V.T @ self.xm - self.actual_values)
        self.nll[0] = sum(np.divide((self.m - self.actual_values)**2,2*(np.diag(self.k)+ self.sigma2))) + sum(np.log(2*np.pi*(np.diag(self.k)+ self.sigma2))/2)
        # self.tracesqrt = np.zeros(m)
        #add tolerance
        for i in range(m):
            Z = scipy.linalg.solve_triangular(L.T, U.T @ Z ,lower=False,check_finite=False) 
            self.xm = -scipy.linalg.solve_triangular(L, U @ self.xm ,lower=True,check_finite=False) + f
            M = sqD * Z
            self.VDmV = self.VDmV + (M.T @ M)
            self.vm = self.m_0ts + self.V.T @ self.xm
            self.m = self.m_0ts + self.vm
            self.k =  self.CXtestXest - self.VDmV
            self.rmse[i+1] = np.linalg.norm(self.vm - self.actual_values)
            self.nll[i+1] = sum(np.divide((self.m - self.actual_values)**2,2*(np.diag(self.k)+ self.sigma2))) + sum(np.log(2*np.pi*(np.diag(self.k)+ self.sigma2))/2)
        pass

    def PLS_CG(self, m, P = None):
        if P is None:
            P = np.eye(self.Xtrdim)
        xm = np.zeros(self.Xtrdim)
        self.time_cg = np.zeros(m)
        self.rmse_cg = np.zeros(m+1)
        self.Dcg = np.eye(self.Xtrdim) - np.eye(self.Xtrdim)
        self.mcg = self.m_0ts + self.V.T @ xm 
        self.C_cg = self.CXtestXest - (self.V.T @ self.Dcg @ self.V)
        self.rmse_cg[0] = np.linalg.norm(self.V.T @ xm - self.actual_values)
        self.nll_cg = np.zeros(m+1)
        self.nll_cg[0] = sum(np.divide((self.mcg - self.actual_values)**2,2*(np.diag(self.C_cg)+self.sigma2))) + sum(np.log(2*np.pi*(np.diag(self.C_cg)+self.sigma2))/2)
        # self.tracesqrt_cg = np.zeros(m)
        #check this
        self.Dcg = np.eye(self.Xtrdim) - np.eye(self.Xtrdim)
        for i in range(m):
            r = self.b - (self.G @ xm)
            s = P @ r
            #check the order of s and r computation
            alpha = np.dot(s,r)
            d = (np.eye(self.Xtrdim) - self.Dcg @ self.G) @ s
            eta = np.dot(s, self.G @ d)
            self.Dcg = self.Dcg + ((1 / eta) * np.outer(d, d))
            xm = xm + alpha / eta * d
            # self.tracesqrt_cg[i] = np.sum(np.sqrt(np.diag(self.CXtestXest- (self.V.T @ self.Dcg @ self.V))))
            self.mcg = self.m_0ts + self.V.T @ xm 
            self.C_cg = self.CXtestXest - (self.V.T @ self.Dcg @ self.V)
            self.rmse_cg[i+1] = np.linalg.norm((self.mcg) - self.actual_values)
            self.nll_cg[i+1] = sum(np.divide((self.mcg - self.actual_values)**2,2*(np.diag(self.C_cg)+self.sigma2))) + sum(np.log(2*np.pi*(np.diag(self.C_cg)+self.sigma2))/2)
        # plt.plot(xm, c= 'blue')
        pass


    def full_post(self):
        self.mgp = self.m_0ts + self.V.T @ np.linalg.solve(self.G,self.b)
        self.kgp = self.CXtestXest - (self.V.T @ np.linalg.inv(self.G) @ self.V)
        self.rmse_full = np.linalg.norm(self.mgp - self.actual_values)
        self.nll_full = sum(np.divide((self.mgp - self.actual_values)**2,2*(np.diag(self.kgp)+ self.sigma2))) + sum(np.log(2*np.pi*(np.diag(self.kgp)+ self.sigma2))/2)
