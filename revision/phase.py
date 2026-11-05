"""Legacy-normalized-time cycles, explicit perturbations, no hidden resampling upgrade."""
import numpy as np
from pglts.signal.cycle import dominant_f0, cycle_boundaries, resample_cycle

def starts_for(amp,pert=None,seed=0):
    pert=pert or {};rng=np.random.default_rng(seed)
    starts=cycle_boundaries(amp,dominant_f0(amp));original=len(starts)
    kind=pert.get('kind','clean');amount=pert.get('amount',0)
    if kind=='boundary_jitter':starts=starts+rng.integers(-int(amount),int(amount)+1,size=len(starts))
    elif kind=='boundary_offset':starts=starts+int(amount)
    elif kind=='phase_offset':
        period=1000/max(dominant_f0(amp),1);starts=starts+int(round(amount*period))
    elif kind=='missed_boundary':
        keep=rng.random(len(starts))>=amount
        if len(keep):keep[[0,-1]]=True
        starts=starts[keep]
    starts=np.unique(np.clip(starts,0,len(amp)))
    return starts,original

def cycles(amp,cpx,B,max_cycles=200,pert=None,seed=0,return_stats=False):
    pert=pert or {};kind=pert.get('kind','clean');amount=pert.get('amount',0)
    a=np.array(amp,float,copy=True);z=np.array(cpx,complex,copy=True);rng=np.random.default_rng(seed)
    if kind=='awgn':
        # Noise is referenced to the prepared micro-Doppler component.
        a+=rng.normal(0,np.std(a)*10**(-amount/20),len(a))
        sigma=np.sqrt(np.mean(np.abs(z)**2)/2)*10**(-amount/20)
        z+=sigma*(rng.normal(size=len(z))+1j*rng.normal(size=len(z)))
    elif kind=='radar_phase_jitter':z*=np.exp(1j*rng.normal(0,amount,len(z)))
    starts,original=starts_for(a,pert,seed)
    freq=np.diff(np.unwrap(np.angle(z)))*1000/(2*np.pi);freq=np.r_[freq,freq[-1]]
    channels=[a,np.abs(z),freq];data=[];rejected=0
    all_pairs=list(zip(starts[:-1],starts[1:]))
    for start,stop in all_pairs[:max_cycles]:
        if stop-start<2:rejected+=1;continue
        v=np.stack([resample_cycle(x,int(start),int(stop),B) for x in channels],1)
        sd=v.std(0,keepdims=True);sd[sd<=0]=1
        v=(v-v.mean(0,keepdims=True))/sd
        if not np.isfinite(v).all():rejected+=1;continue
        data.append(v)
    x=np.asarray(data,dtype=np.float32).reshape(-1,B,3)
    stats={'original_boundaries':original,'detected_boundaries':len(starts),'candidate_cycles':max(len(starts)-1,0),'used_cycles':len(x),'invalid_cycles':rejected,'segment_rejected':int(len(x)<2)}
    return (x,stats) if return_stats else x
