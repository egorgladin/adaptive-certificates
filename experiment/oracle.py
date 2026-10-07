"""Exact full-data profiled kernel hinge SVM / standard scaled nu-SVR."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
from pathlib import Path
import numpy as np
import torch


class KernelProblem:
    def __init__(self,dataset):
        path = Path('output/data')/dataset
        self.meta = json.loads((path/'metadata.json').read_text())
        self.dataset,self.problem = dataset,self.meta['problem']
        self.lam,self.nu,self.d = self.meta['lam'],self.meta['nu'],self.meta['d']
        self.B = torch.tensor(np.load(path/'B.npy'),device='cuda')
        self.y = torch.tensor(np.load(path/'y.npy'),device='cuda')
        self.positive = int((self.y>0).sum()) if self.problem=='svm' else 0
        self.m = len(self.y)
        if self.problem=='nusvr':
            half = self.nu*self.m/2
            self.tail_count,self.fraction = int(np.floor(half)),float(half%1)

    def oracle(self,x):
        prediction = self.B@x
        if self.problem=='nusvr':
            residual = self.y-prediction
            order = torch.argsort(residual,stable=True)
            count,fraction = self.tail_count,self.fraction
            s = torch.zeros_like(residual)
            s[order[:count]] = 1/self.m
            if count:
                s[order[-count:]] = -1/self.m
            if fraction>1e-13:
                s[order[count]] += fraction/self.m
                s[order[self.m-count-1]] -= fraction/self.m
                intercept = (residual[order[count]]+residual[order[self.m-count-1]])/2
            else:
                intercept = (residual[order[count-1]]+residual[order[self.m-count]])/2
            h = -torch.dot(s,residual)
        else:
            # At b=-infinity derivative is -n_positive/n; each sorted
            # breakpoint increases derivative by1/n. Middle interval minimizes h.
            breakpoints = self.y-prediction
            values = torch.sort(breakpoints,stable=True).values
            intercept = (values[self.positive-1]+values[self.positive])/2
            residual = 1-self.y*(prediction+intercept)
            active = (residual>0).to(x.dtype)
            zero = (residual==0)
            mass = torch.dot(self.y,active)
            positive_zero = zero & (self.y>0)
            negative_zero = zero & (self.y<0)
            # Fractional coefficients at tied breakpoints make sum(s)=0.
            active += positive_zero*torch.clamp(-mass,min=0)/torch.clamp(positive_zero.sum(),min=1)
            active += negative_zero*torch.clamp(mass,min=0)/torch.clamp(negative_zero.sum(),min=1)
            s = -self.y*active/self.m
            h = torch.clamp(residual,min=0).mean()
        a = self.B.T@s
        g = a+self.lam*x
        F = h+self.lam*torch.dot(x,x)/2
        b = h-torch.dot(a,x)
        packet = torch.stack((F,b,torch.dot(g,g),intercept,torch.dot(x,x),torch.sum(x),torch.sum(s)))
        return F,g,a,packet

    def score(self,x,intercept,split='test'):
        path = Path('output/data')/self.dataset
        matrix = torch.tensor(np.load(path/('B'+split+'.npy')),device='cuda')
        target = torch.tensor(np.load(path/('y'+split+'.npy')),device='cuda')
        prediction = matrix@x+intercept
        if self.problem=='svm':
            return float(((prediction>=0)==(target>0)).to(x.dtype).mean())
        return float(torch.sqrt(torch.mean((prediction-target)**2)))
