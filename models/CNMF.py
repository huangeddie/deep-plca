import numpy as np
from collections import defaultdict

class cnmf2D(object):
    def __init__(self, input_size, kernel_size, nKernel, nInput):
        self.iSize_ = input_size
        self.kSize_ = kernel_size
        self.nKernel_ = nKernel
        self.nInput_ = nInput
        self.pwk, self.ph = self._uniform_init(input_size, kernel_size, nKernel, nInput)
        self.rInd_, self.h2w_ = self._get_R_ind()
        self.pz = np.array([1/nKernel]*nKernel).reshape(-1,1)
        self.recon = np.zeros((self.iSize_[0],self.iSize_[1], nInput))

    def _uniform_init(self, input_size, kernel_size, nKernel, nInput):
        wk = np.random.uniform(size = (kernel_size[0], kernel_size[1], nKernel))
        wk /= (kernel_size[0] * kernel_size[1]) # TODO might cause numerical instability
        self.hx_ = input_size[0] - kernel_size[0] + 1
        self.hy_ = input_size[1] - kernel_size[1] + 1
        h = np.random.uniform(high = 1/(self.hx_ * self.hy_), size = (self.hx_, self.hy_, nKernel, nInput))
        h /= (self.hx_*self.hy_)

        return wk, h

    def _get_R_ind(self):
        w2h = defaultdict(list)
        h2w = defaultdict(list)
        kx = self.kSize_[0]
        ky = self.kSize_[1]

        for i in range(self.iSize_[0]):
            for j in range(self.iSize_[1]):
                for m in range(kx):
                    for n in range(ky):
                        hx = i-kx+m+1
                        hy = j-ky+n+1

                        if 0<=hx<self.hx_ and 0<=hy<self.hy_:
                            w2h[i,j,'h'].append((hx,hy))
                            w2h[i,j,'k'].append((kx-m-1,ky-n-1))
                            h2w[hx,hy,'w'].append((i,j))
                            h2w[hx,hy,'pos'].append(len(w2h[i,j,'h']) - 1)

        return w2h,h2w

    def _E_step(self):
        # update self.R_
        self.R_ = defaultdict(list) # R[i,j] = array(nAdj, nKernel, nInput)
        for i in range(self.iSize_[0]):
            for j in range(self.iSize_[1]):
                d = 0
                for h,k in zip(self.rInd_[i,j,'h'], self.rInd_[i,j,'k']):
                    hx,hy = h
                    kx,ky = k
                    n = self.pwk[kx,ky,:].reshape(-1,1) * self.pz * self.ph[hx,hy,:,:]# nKern *  nKern * (nKernel, nInput)
                    d += np.sum(n, axis = 0)
                    self.R_[i,j].append(n)
                #for ind in np.where(d == 0)[0]:
                #    d[ind] = 1e-6
                #    self.R_[i,j][0][0,ind] = 1e-6
                self.recon[i,j,:] = d
                self.R_[i,j] = np.array(self.R_[i,j])/(d) #nAdj, nKern, nInput

    def _M_step(self, x, curr_iter, nIter):
        self._update_pz(x)
        self._update_pwk(x, curr_iter, nIter)
        self._update_ph(x, curr_iter, nIter)

    def _update_pz(self, x):
        pz = 0
        for w in self.R_.keys():
            i,j = w
            pz += np.sum(x[:,i,j] * self.R_[w], axis = (0,2)).reshape(-1,1)

        self.pz = pz/self.nInput_

    def _update_pwk(self, x, curr_iter, nIter):
        """
        x: ndarray(nInput, input_size[0], input_size[1])
        """
        if curr_iter >= nIter/2:
            curr_iter = -1
        p = np.linspace(0.7,1,int(nIter/2))[curr_iter]
        currKern = np.zeros((self.kSize_[0], self.kSize_[1], self.nKernel_))
        for i in range(self.iSize_[0]):
            for j in range(self.iSize_[1]):
                for ind,k in enumerate(self.rInd_[i,j,'k']):
                    kx,ky = k
                    tmp = self.R_[i,j][ind,:,:] #nKern * nInput
                    tmp = tmp * x[:,i,j]
                    tmp = np.sum(tmp, axis = 1) #nKern
                    currKern[kx,ky,:] += tmp

        currKern /= self.pz.flatten()
        currKern = np.power(currKern, p)
        self.pwk[:,:,:] = currKern/np.sum(currKern, axis = (0,1))
        #self.pwk[:,:,:] = currKern/self.pz.flatten()

    def _update_ph(self, x, curr_iter, nIter):
        if curr_iter >= nIter/2:
            curr_iter = 0
        else:
            curr_iter = -curr_iter-1
        p = np.linspace(1,2,int(nIter/2))[curr_iter]
        for hx in range(self.hx_):
            for hy in range(self.hy_):
                currH = np.zeros((self.nKernel_, self.nInput_))
                wInd = self.h2w_[hx,hy,'w']
                pos = self.h2w_[hx,hy,'pos']
                for w,p in zip(wInd, pos):
                    i,j = w
                    m = x[:,i,j].flatten()
                    currH += self.R_[i,j][p,:,:] * m #(nKern, nInput) * (nInput,1)

                self.ph[hx,hy,:,:] = currH

        self.ph = self.ph/np.sum(self.ph, axis = (0,1))
        self.ph = np.power(self.ph, 1.2) + 1e-12
        self.ph = self.ph/np.sum(self.ph, axis = (0,1))


    def train(self,x, nIter):
        for i in range(nIter):
            self._E_step()
            self._M_step(x, i, nIter)
            print('Done with iteration {}'.format(i))

    def get_recon(self):
        return self.recon
