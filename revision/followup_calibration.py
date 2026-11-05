"""Subject-aggregated synthetic calibration, not biological phase validation.

The exact measured statistic is mean_segments(mean_channels(L1(mean_cycles(Gamma)))).
IID-bin nulls support uniform-bin randomization. Circularly smoothed nulls support
cyclic shifts; uniform-bin rejection on these is a diagnostic, not type-I error.
Overlap stress reuses physical cycle IDs in adjacent segments. The legacy-style
occurrence randomization then breaks this dependence; coherent randomization
shares draws by cycle ID. Neither is asserted to model real radar exactly.
"""
import itertools
import numpy as np
from scipy.stats import binomtest
from .order import residual
from .storage import csv_write,digest


def memberships(x,states=4):
    """Fixed two-adjacent-state triangular memberships; no test-data fitting."""
    q=np.clip(np.asarray(x,float),0,1)*(states-1)
    left=np.minimum(np.floor(q).astype(int),states-2);frac=q-left
    eye=np.eye(states)
    return eye[left]*(1-frac[...,None])+eye[left+1]*frac[...,None]


def subject_stat(mu,counts=None):
    # segment x cycles x bins x channels x states
    gamma=np.stack([residual(segment) for segment in mu])
    if counts is None: averaged=gamma.mean(1)
    else:
        mask=np.arange(mu.shape[1])[None,:]<np.asarray(counts)[:,None]
        averaged=(gamma*mask[:,:,None,None,None]).sum(1)/np.asarray(counts)[:,None,None,None]
    return float(np.abs(averaged).sum((-2,-1)).mean(-1).mean())


def layout(segments,cycles,overlap):
    step=max(1,cycles//2) if overlap else cycles
    return np.arange(cycles)[None,:]+step*np.arange(segments)[:,None]


def generate(cfg,case,rng):
    S,C,B=case['segments'],case['cycles'],case['bins'];ids=layout(S,C,case['overlap'])
    counts=case.get('cycle_counts')
    if counts is not None:
        # Independent empirical segments: preserve each exact retained cycle count.
        if case['overlap']:raise ValueError('Exact empirical ragged profiles are independent; overlap is a separate stress model.')
        ids=np.full((S,max(counts)),0,int);offset=0
        for j,nc in enumerate(counts):ids[j,:nc]=np.arange(offset,offset+nc);offset+=nc
        n=offset
    else:n=int(ids.max())+1
    channels=cfg['channels']
    # Correlated channels, varying cycles, but independent positions under IID null.
    base=rng.uniform(size=(n,B,1));x=.65*base+.35*rng.uniform(size=(n,B,channels))
    kind=case['generator'];strength=case['strength']
    if kind=='smooth':
        # Circular convolution preserves stationarity under a random cyclic shift.
        x=(np.roll(x,1,axis=1)+2*x+np.roll(x,-1,axis=1))/4
    else:
        # Structured alternative has independently varying cycle values and a
        # shared ascending orientation on a strength-controlled fraction of cycles.
        x=np.sort(x,axis=1)
    mu=memberships(x,cfg['states'])
    if kind=='smooth':
        index=(np.arange(B)[None,:]+rng.integers(B,size=(n,1)))%B
    else:index=np.argsort(rng.random((n,B)),axis=1)
    if strength:
        aligned=rng.random(n)<strength;index[aligned]=np.arange(B)
    mu=np.take_along_axis(mu,index[:,:,None,None],axis=1)
    return mu,ids


def draws(mu,ids,group,coupling,rng,batch):
    n,B,R,A=mu.shape;S,C=ids.shape
    shape=(batch,n,B) if coupling=='coherent_cycle_id' else (batch,S,C,B)
    if group=='uniform_bins':index=np.argsort(rng.random(shape),axis=-1)
    elif group=='cyclic_shift':index=(np.arange(B)+rng.integers(B,size=shape[:-1]+(1,)))%B
    else:raise ValueError('Unknown randomization group: '+group)
    if coupling=='coherent_cycle_id':
        perm=np.take_along_axis(np.broadcast_to(mu,(batch,n,B,R,A)),index[:,:,:,None,None],axis=2)
        return perm[:,ids]
    # This matches independent segment-occurrence permutations in measured_order.
    observed=np.broadcast_to(mu[ids],(batch,S,C,B,R,A))
    return np.take_along_axis(observed,index[:,:,:,:,None,None],axis=3)


def expected_mean(mu,counts=None):
    # segment x cycle x bin x channel x state. Invariant under both groups.
    L=mu.shape[2];O=mu.mean(2)
    E=(L*L*np.einsum('scra,scrz->scraz',O,O,optimize=True)-np.einsum('scira,scirz->scraz',mu,mu,optimize=True))/(L*(L-1))
    if counts is None:return E.mean(1)
    mask=np.arange(mu.shape[1])[None,:]<np.asarray(counts)[:,None]
    return (E*mask[:,:,None,None,None]).sum(1)/np.asarray(counts)[:,None,None,None]


def batch_stats(x,counts=None,expected=None):
    # Sum adjacent products over cycles before creating the small state matrix.
    batch,S,C,L,R,A=x.shape
    left=x[:,:,:,:-1].transpose(0,1,4,2,3,5).reshape(batch,S,R,C*(L-1),A)
    right=x[:,:,:,1:].transpose(0,1,4,2,3,5).reshape(batch,S,R,C*(L-1),A)
    if counts is not None:
        mask=(np.arange(C)[None,:]<np.asarray(counts)[:,None])
        mask=np.repeat(mask,L-1,axis=1)
        left=left*mask[None,:,None,:,None]
    denom=(np.full(S,C) if counts is None else np.asarray(counts))*(L-1)
    M=np.matmul(left.swapaxes(-1,-2),right)/denom[None,:,None,None,None]
    if expected is None:expected=np.stack([expected_mean(mu,counts) for mu in x])
    else:expected=expected[None]
    return np.abs(M-expected).sum((-2,-1)).mean(-1).mean(-1)


def single(cfg,case,rep):
    seed=int(digest({'seed':cfg['seed'],'case':case,'rep':rep})[:16],16)
    rng=np.random.default_rng(seed);mu,ids=generate(cfg,case,rng);obs=subject_stat(mu[ids],case.get('cycle_counts'));expected=expected_mean(mu[ids],case.get('cycle_counts'));exceed=0;Q=cfg['null_permutations']
    for start in range(0,Q,cfg['draw_batch']):
        values=batch_stats(draws(mu,ids,case['group'],case['coupling'],rng,min(cfg['draw_batch'],Q-start)),case.get('cycle_counts'),expected)
        exceed+=int(np.sum(values>=obs-1e-12))
    p=(1+exceed)/(Q+1)
    reference_supported=(case['generator']=='iid' or case['group']=='cyclic_shift') and (not case['overlap'] or case['coupling']=='coherent_cycle_id')
    supported=reference_supported and case['strength']==0
    meaning='power_for_this_alternative' if case['strength']>0 else ('false_positive_rate_under_supported_null' if supported else 'rejection_rate_under_null_mismatch')
    return {**case,'rep':rep,'statistic':obs,'p_value':p,'reject':int(p<=cfg['alpha']),'meaning':meaning,'exchangeability_supported_by_generator':supported,'reference_null_group_supported':reference_supported,'null_draws':Q,'data_mode':'synthetic_aggregated_control'}


def cases(cfg):
    result=[]
    for profile,scenario,group in itertools.product(cfg['profiles'],cfg['scenarios'],cfg['groups']):
      for B in profile.get('bins',cfg['bins']):
        for coupling in (cfg['overlap_couplings'] if profile.get('overlap',False) else ['independent_occurrence']):
            result.append({'profile':profile['name'],'segments':profile['segments'],'cycles':profile['cycles'],'overlap':profile.get('overlap',False),'bins':B,'generator':scenario['generator'],'strength':scenario['strength'],'group':group,'coupling':coupling,**({'cycle_counts':profile['cycle_counts']} if 'cycle_counts' in profile else {})})
    return result


def run(cfg,store,folder):
    rows=[];summaries=[]
    for case in cases(cfg):
        case_id=digest(case)[:12];subset=[]
        for rep in range(cfg['repetitions']):
            r=store.task(f'aggregation/{case_id}/{rep:05d}',lambda case=case,rep=rep:single(cfg,case,rep));subset.append(r);rows.append(r)
        k=sum(r['reject'] for r in subset);ci=binomtest(k,len(subset)).proportion_ci(confidence_level=.95)
        summaries.append({**case,'repetitions':len(subset),'rejection_rate':k/len(subset),'lo':ci.low,'hi':ci.high,'meaning':subset[0]['meaning'],'exchangeability_supported_by_generator':subset[0]['exchangeability_supported_by_generator']})
        csv_write(folder/'aggregation_calibration_raw.csv',rows);csv_write(folder/'aggregation_calibration_summary.csv',summaries)
    return {'cases':len(summaries),'repetitions':len(rows)}
