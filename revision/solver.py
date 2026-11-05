"""Sparse-group softmax with weighted loss and backtracking, free intercept.
Reuses legacy block prox; fixes weighted/intercept Lipschitz bound, true iteration
count, convergence diagnostics, and finite-objective checks.
"""
import numpy as np
from pglts.models.sparse_group import _prox, _softmax
from scipy.special import logsumexp

class StableSparseGroup:
    def __init__(self,l1=.001,lg=.0005,max_iter=1000,tol=1e-6,class_weight="balanced"):
        self.l1=l1;self.lg=lg;self.max_iter=max_iter;self.tol=tol;self.class_weight=class_weight
    def fit(self,X,y,groups):
        X=np.asarray(X,float);y=np.asarray(y);self.classes_,idx=np.unique(y,return_inverse=True)
        n,d=X.shape;k=len(self.classes_)
        if k<2:raise ValueError('Training fold has fewer than two classes.')
        Y=np.eye(k)[idx];sw=n/(k*np.bincount(idx))[idx] if self.class_weight=="balanced" else np.ones(n)
        g=np.asarray(groups if groups is not None else [])
        if g.size not in (0,d):raise ValueError('Group count differs from feature count.')
        blocks=[np.flatnonzero(g==x) for x in np.unique(g)]
        W=np.zeros((d,k));b=np.zeros(k);Z=W.copy();c=b.copy();momentum=1.;L=1.
        def smooth(w,inter):
            z=X@w+inter
            return float(np.mean(sw*(logsumexp(z,axis=1)-z[np.arange(n),idx])))
        def penalty(w):
            return self.l1*np.abs(w).sum()+self.lg*sum(np.sqrt(len(ix))*np.linalg.norm(w[ix]) for ix in blocks)
        prev=smooth(W,b)+penalty(W);self.converged_=False
        for iteration in range(1,self.max_iter+1):
            P=_softmax(X@Z+c);res=(P-Y)*sw[:,None];gw=X.T@res/n;gb=res.mean(0);base=smooth(Z,c)
            for _ in range(80):
                wn=_prox(Z-gw/L,1/L,self.l1,self.lg,blocks);bn=c-gb/L
                dw=wn-Z;db=bn-c;fn=smooth(wn,bn)
                upper=base+(gw*dw).sum()+gb@db+L/2*((dw*dw).sum()+db@db)
                if np.isfinite(fn) and fn<=upper+1e-10:break
                L*=2
            else:raise FloatingPointError('Backtracking failed.')
            obj=fn+penalty(wn)
            if obj>prev+1e-10:
                # Monotone restart; recompute the next step at the last iterate.
                Z=W.copy();c=b.copy();momentum=1.;continue
            relative=np.sqrt(((wn-W)**2).sum()+((bn-b)**2).sum())/(1+np.linalg.norm(W)+np.linalg.norm(b))
            t=(1+np.sqrt(1+4*momentum*momentum))/2
            Z=wn+(momentum-1)/t*(wn-W);c=bn+(momentum-1)/t*(bn-b)
            W,b,momentum,prev=wn,bn,t,obj
            if relative<self.tol:self.converged_=True;break
        self.coef_=W.T;self.intercept_=b;self.n_iter_=iteration;self.objective_=float(prev)
        if not np.isfinite(self.coef_).all():raise FloatingPointError('Non-finite fitted coefficients.')
        return self
    def decision_function(self,X):return X@self.coef_.T+self.intercept_
    def predict(self,X):return self.classes_[self.decision_function(X).argmax(1)]
