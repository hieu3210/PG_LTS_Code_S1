"""Train-only linguistic maps and matched-input per-cycle controls."""
import hashlib,json
from functools import lru_cache
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from pglts.linguistic.semantize import make_semantizer
from pglts.linguistic.descriptors import DescriptorBank,static_statistics
from pglts.signal.features import PGLTS_CORE
from .solver import StableSparseGroup
from .phase import cycles

@lru_cache(maxsize=8192)
def _cached_cycles(identity,B,max_cycles,amp_bytes,cpx_bytes):
    return cycles(np.frombuffer(amp_bytes,np.float32),np.frombuffer(cpx_bytes,np.complex64),B,max_cycles)

def get_cycles(frame,B,max_cycles):
    return _cached_cycles(frame['id'],B,max_cycles,frame['amp'].tobytes(),frame['cpx'].tobytes())

def permute_block(block,seed,identity):
    # One independent within-cycle permutation, shared by all channels.
    value=int.from_bytes(hashlib.sha256(f'{seed}:{identity}'.encode()).digest()[:8],'little')
    rng=np.random.default_rng(value)
    indices=np.argsort(rng.random(block.shape[:2]),axis=1)
    return np.take_along_axis(block,indices[:,:,None],axis=1)

class Model:
    def __init__(self,candidate,options,perm_seed=None):
        self.c=dict(candidate);self.options=options;self.perm_seed=perm_seed;self.parts=[];self.names=[];self.groups=[]
    def _cycle_blocks(self,frames,pert=None,seed=0):
        out=[]
        for f in frames:
            if pert:
                val=int.from_bytes(hashlib.sha256(f"{seed}:{f['id']}".encode()).digest()[:4],'little')
                x=cycles(f['amp'],f['cpx'],self.c['B'],self.options['max_cycles'],pert,val)
            else:x=get_cycles(f,self.c['B'],self.options['max_cycles'])
            if self.perm_seed is not None:x=permute_block(x,self.perm_seed,f['id'])
            out.append(x)
        return out
    def _describe(self,frames,fit=False,pert=None,seed=0):
        view=self.c['view'];method=self.c['method'];win=np.array([f['window'] for f in frames]);pieces=[];names=[];groups=[]
        if method=='svm':
            x,names=static_statistics(win);return x,names,['aggregate']*x.shape[1]
        has_win=view in ('occupancy','combined','full_window')
        has_cycle=view in ('phase_occ','phase_ordered','combined')
        if method in ('rocket','multirocket','linear'):
            has_win=view=='combined';has_cycle=True
            if has_win:
                x,n=static_statistics(win);pieces.append(x);names.extend('window:'+z for z in n);groups+=['window']*x.shape[1]
        elif has_win:
            if fit:
                self.win_sem=make_semantizer(self.c['partition'],n_states=self.c['states']).fit(win)
                self.win_bank=DescriptorBank(families=('O','P','M','C') if view=='full_window' else ('O',),feature_names=PGLTS_CORE,words=self.win_sem.words)
            x,n,g=self.win_bank.transform(self.win_sem.transform(win));pieces.append(x);names+=['window:'+z for z in n];groups+=['window:'+z for z in g]
        if has_cycle:
            blocks=self._cycle_blocks(frames,pert,seed)
            if any(len(x)<2 for x in blocks):raise ValueError('Cycle rejection: filter cohort explicitly before prediction.')
            if method=='linear':
                x=np.stack([np.r_[b.mean(0).ravel(),b.std(0).ravel()] for b in blocks])
                n=[f'cycle:{i}' for i in range(x.shape[1])];g=['cycle']*x.shape[1]
            elif method in ('rocket','multirocket'):
                # Fixed right-edge extension to >=10 supports short-cycle kernels;
                # it adds no measured samples. Fit only on outer/inner training cycles.
                def tensor(b):
                    t=b.transpose(0,2,1)
                    return np.pad(t,((0,0),(0,0),(0,max(0,10-t.shape[2]))),mode='edge').astype(np.float32)
                if fit:
                    from aeon.transformations.collection.convolution_based import Rocket,MultiRocket
                    cls=Rocket if method=='rocket' else MultiRocket
                    self.rocket=cls(n_kernels=self.c['kernels'],n_jobs=self.options['threads'],random_state=self.c['seed'])
                    raw=np.concatenate([tensor(b) for b in blocks]);rng=np.random.default_rng(self.c['seed'])
                    if len(raw)>self.options['rocket_fit_cycles']:raw=raw[rng.choice(len(raw),self.options['rocket_fit_cycles'],replace=False)]
                    self.rocket.fit(raw)
                # Do not allocate total_cycles x total_features for the corpus.
                x=np.stack([np.asarray(self.rocket.transform(tensor(b))).mean(0) for b in blocks])
                n=[f'{method}:{i}' for i in range(x.shape[1])];g=['cycle']*x.shape[1]
            else:
                if fit:
                    flat=np.concatenate(blocks)
                    self.cyc_sem=make_semantizer(self.c['partition'],n_states=self.c['states']).fit(flat)
                    self.cyc_bank=DescriptorBank(families=('O',) if view=='phase_occ' else ('O','P','M'),pairs=(),feature_names=('waveform','envelope','inst_freq'),words=self.cyc_sem.words)
                desc=[]
                for b in blocks:
                    d,n,g=self.cyc_bank.transform(self.cyc_sem.transform(b));desc.append(d.mean(0))
                x=np.stack(desc);n=['cycle:'+z for z in n];g=['cycle:'+z for z in g]
            pieces.append(x);names+=n;groups+=g
        return np.concatenate(pieces,axis=1).astype(float),names,groups
    def fit(self,frames):
        x,self.names,self.groups=self._describe(frames,True);self.scaler=StandardScaler().fit(x);z=self.scaler.transform(x);y=np.array([f['y'] for f in frames])
        if self.c['method']=='linguistic':
            self.clf=StableSparseGroup(self.c['l1'],self.c['lg'],self.options['max_iter'],self.options['tol']).fit(z,y,self.groups)
        elif self.c['method']=='svm':self.clf=SVC(C=self.c['C'],kernel='rbf',class_weight='balanced').fit(z,y)
        else:self.clf=LogisticRegression(C=self.c['C'],class_weight='balanced',max_iter=self.options['max_iter'],tol=self.options['tol']).fit(z,y)
        self.classes_=self.clf.classes_;self.train_abs=np.abs(z).mean(0)
        self.converged=bool(getattr(self.clf,'converged_',True))
        if hasattr(self.clf,'n_iter_') and self.c['method'] not in ('linguistic','svm'):
            self.converged=bool(np.max(self.clf.n_iter_)<self.options['max_iter'])
        if not self.converged and self.options['require_convergence']:raise RuntimeError('Solver did not converge; increase max_iter in a NEW run config.')
        return self
    def transform(self,frames,pert=None,seed=0):return self.scaler.transform(self._describe(frames,False,pert,seed)[0])
    def scores(self,frames,pert=None,seed=0):
        z=self.transform(frames,pert,seed);s=self.clf.decision_function(z)
        if s.ndim==1:s=np.c_[-s/2,s/2]
        return s
    def predict(self,frames,pert=None,seed=0):return self.classes_[self.scores(frames,pert,seed).argmax(1)]
    def active(self):
        if not hasattr(self.clf,'coef_'):return len(self.names)
        return int((np.max(np.abs(self.clf.coef_),axis=0)>1e-10).sum())
