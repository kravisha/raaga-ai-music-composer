"""Numba version of the pinned upstream monotonic_align/core.pyx recurrence.

The isolated Windows setup lacks MSVC. This adapter leaves upstream source
unchanged; its equivalence is checked before any model step. No fallback is
silently selected by the working inference environment.
"""
import numpy as np
import torch
from numba import njit

@njit(cache=False)
def _paths(values, ys, xs):
    paths=np.zeros(values.shape,np.int32)
    for b in range(values.shape[0]):
        ty,tx=ys[b],xs[b]
        for y in range(ty):
            for x in range(max(0,tx+y-ty),min(tx,y+1)):
                cur=-1e9 if x==y else values[b,y-1,x]
                prev=(0. if y==0 else -1e9) if x==0 else values[b,y-1,x-1]
                values[b,y,x]+=max(prev,cur)
        index=tx-1
        for y in range(ty-1,-1,-1):
            paths[b,y,index]=1
            if index!=0 and y!=0 and (index==y or values[b,y-1,index]<values[b,y-1,index-1]):index-=1
    return paths

def maximum_path(scores,mask):
    values=scores.detach().cpu().numpy().astype(np.float32,copy=True)
    m=mask.detach().cpu().numpy()
    if values.ndim!=3 or m.shape!=values.shape or not np.isfinite(values).all():raise ValueError('invalid alignment scores')
    ys=m.sum(axis=1)[:,0].astype(np.int32);xs=m.sum(axis=2)[:,0].astype(np.int32)
    for b,(ty,tx) in enumerate(zip(ys,xs)):
        expected=np.zeros(m.shape[1:],dtype=m.dtype);expected[:ty,:tx]=1
        if tx<1 or ty<tx or not np.array_equal(expected,m[b]):raise ValueError('alignment mask must be rectangular with enough audio frames')
    return torch.from_numpy(_paths(values,ys,xs)).to(device=scores.device,dtype=scores.dtype)
