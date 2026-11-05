"""Finite-sample calibration of the v1.2 recording-shared tests.
Synthetic controls only. Exact empirical counts, not reconstructed radar phase.
The cycle-mixture generator permits an exact moment reduction (tested against
explicit cycles) so 999 null draws do not require resampling all cycle arrays.
"""
from pathlib import Path
import argparse,csv,itertools,json,math,traceback,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits
from . import ROOT
from .storage import Store,run_lock,digest,sha,json_write,csv_write
from .radar_order import patterns
VERSION='1.3.0'
GROUPS=['recording_block_shared','recording_cyclic_shared']
SCENARIOS=['uniform_shared','block_shared_smooth','cyclic_shared_smooth','aligned_035','aligned_080']

def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def profiles_from_manifest(path):
    rows=read_csv(path);keys=set();profiles=[]
    for r in rows:
        key=(r['cohort'],r['id'])
        if key in keys:raise ValueError('Duplicate cohort/segment')
        keys.add(key)
    for cohort,subject in sorted({(r['cohort'],r['subject']) for r in rows}):
        if cohort not in ['grid0','grid1000']:continue
        rr=[r for r in rows if (r['cohort'],r['subject'])==(cohort,subject)]
        bins={int(r['bins']) for r in rr}
        if len(bins)!=1:raise ValueError('Inconsistent B within subject')
        records=[]
        for rec in sorted({r['recording'] for r in rr}):
            sub=sorted([r for r in rr if r['recording']==rec],key=lambda r:r['id'])
            counts=[int(r['cycles']) for r in sub]
            if min(counts)<2:raise ValueError('Invalid cycle count')
            # Lattice assertion is separate from filtered statistical independence.
            starts=sorted(int(r['id'].rsplit(':',2)[1]) for r in sub)
            if any(b-a<2000 for a,b in zip(starts,starts[1:])):raise ValueError('Segments overlap in grid cohort')
            records.append({'recording':rec,'ids':[r['id'] for r in sub],'cycle_counts':counts})
        profiles.append({'cohort':cohort,'subject':subject,'bins':bins.pop(),'segments':len(rr),'records':records})
    if not profiles:raise ValueError('No grid profiles')
    return profiles

def memberships(x,A):
    q=np.clip(x,0,1)*(A-1);i=np.minimum(q.astype(int),A-2);f=q-i;e=np.eye(A)
    return e[i]*(1-f[...,None])+e[i+1]*f[...,None]

def orbit_tables(path,qbar,index,w):
    """Exact mean-cycle residual for mu_c,b = w path_b + (1-w) q_c.
    q_c is constant across bins; its quadratic terms cancel in M-E.
    Returns unweighted per-segment statistics for every shared orientation.
    """
    path=np.asarray(path,float);qbar=np.asarray(qbar,float);B=path.shape[1]
    O=path.mean(1)
    E=(B*B*np.einsum('sra,srz->sraz',O,O)-np.einsum('sbra,sbrz->sraz',path,path))/(B*(B-1))
    p=path[:,np.asarray(index)].transpose(1,0,2,3,4) # P,S,B,R,A
    M=np.einsum('psbra,psbrz->psraz',p[:,:,:-1],p[:,:,1:])/(B-1)
    dl=p[:,:,:-1].mean(2)-O;dr=p[:,:,1:].mean(2)-O
    cross=np.einsum('psra,srz->psraz',dl,qbar)+np.einsum('sra,psrz->psraz',qbar,dr)
    g=w*w*(M-E)+w*(1-w)*cross
    return np.abs(g).sum((-2,-1)).mean(-1) # P,S

def generate_record(counts,B,R,A,scenario,rng,w):
    S=len(counts)
    # Shared recording template + segment residual: correlated segments/channels.
    rec=rng.uniform(.05,.95,(1,B,1));x=.7*rec+.3*rng.uniform(.05,.95,(S,B,R))
    if scenario in ['block_shared_smooth','aligned_035','aligned_080']:
        x=np.sort(x,axis=1)
    elif scenario=='cyclic_shared_smooth':
        x=(np.roll(x,1,axis=1)+2*x+np.roll(x,-1,axis=1))/4
    path=memberships(x,A)
    qrecord=rng.dirichlet(np.ones(A),size=R)
    qbar=[]
    # Exact cycle counts affect the sampled cycle-noise mean, not a padded proxy.
    for n in counts:qbar.append(.7*qrecord+.3*rng.dirichlet(np.ones(A),size=(n,R)).mean(0))
    qbar=np.array(qbar)
    if scenario=='uniform_shared':orient=rng.permutation(B)
    elif scenario=='block_shared_smooth':orient=np.array(patterns(B,GROUPS[0]))[rng.integers(math.factorial(B//2))]
    elif scenario=='cyclic_shared_smooth':orient=(np.arange(B)+rng.integers(B))%B
    else:
        strength={'aligned_035':.35,'aligned_080':.80}[scenario]
        orient=np.arange(B) if rng.random()<strength else rng.permutation(B)
    return path[:,orient],qbar

def null_result(tables,Q,rng,exact_limit):
    obs=float(sum(t[0] for t in tables));size=math.prod(len(t) for t in tables)
    if size<=exact_limit:
        values=np.array([0.])
        for t in tables:values=(values[:,None]+t).ravel()
        k=int((values>=obs-1e-12).sum());p=k/len(values);method='exact';draws=len(values)
    else:
        values=sum(t[rng.integers(len(t),size=Q)] for t in tables)
        k=int((values>=obs-1e-12).sum());p=(1+k)/(Q+1);method='monte_carlo';draws=Q
    return {'statistic':obs,'p_value':float(p),'exceedances':k,'null_draws':draws,'method':method,'group_cardinality':str(size),'p_resolution':1/size if method=='exact' else 1/(Q+1)}

def one_rep(profile,cfg,rep):
    rows=[];B=profile['bins'];N=profile['segments'];R=cfg['channels'];A=cfg['states']
    for scenario in cfg['scenarios']:
        seed=int(digest({'seed':cfg['seed'],'profile':profile,'scenario':scenario,'rep':rep})[:16],16)
        rng=np.random.default_rng(seed);tables={g:[] for g in GROUPS}
        for rec in profile['records']:
            path,qbar=generate_record(rec['cycle_counts'],B,R,A,scenario,rng,cfg['path_weight'])
            for group in GROUPS:
                t=orbit_tables(path,qbar,patterns(B,group),cfg['path_weight']).sum(1)/N
                tables[group].append(t)
        for group in GROUPS:
            supported=scenario=='uniform_shared' or (scenario=='block_shared_smooth' and group==GROUPS[0]) or (scenario=='cyclic_shared_smooth' and group==GROUPS[1])
            meaning='supported_null' if supported else 'specified_alternative' if scenario.startswith('aligned') else 'null_mismatch_diagnostic'
            null_seed=int(digest({'seed':seed,'group':group})[:16],16)
            result=null_result(tables[group],cfg['null_permutations'],np.random.default_rng(null_seed),cfg['exact_limit'])
            rows.append({'cohort':profile['cohort'],'subject':profile['subject'],'bins':B,'segments':N,'recordings':len(profile['records']),'cycles':sum(sum(r['cycle_counts']) for r in profile['records']),'scenario':scenario,'group':group,'rep':rep,'seed':seed,'null_seed':null_seed,'meaning':meaning,'reject':int(result['p_value']<=cfg['alpha']),**result})
    return rows

def build_chunk(profile,cfg,start,stop):
    with threadpool_limits(limits=1):return [r for rep in range(start,stop) for r in one_rep(profile,cfg,rep)]

def commit_chunk(folder,key,profile,cfg,start,stop):
    # Each key has one scheduled owner. Top-level run lock excludes other runners.
    store=Store(Path(folder)/'checkpoints')
    rows=store.task(key,lambda:build_chunk(profile,cfg,start,stop))
    return key,rows,store.completed,store.reused

def summarize(rows,folder,alpha):
    by={}
    for r in rows:by.setdefault(tuple(r[k] for k in ['cohort','subject','scenario','group','meaning']),[]).append(r)
    summary=[]
    for key,rr in sorted(by.items()):
        k=sum(int(r['reject']) for r in rr);ci=binomtest(k,len(rr)).proportion_ci()
        summary.append({**dict(zip(['cohort','subject','scenario','group','meaning'],key)),'repetitions':len(rr),'rejects':k,'rejection_rate':k/len(rr),'lo':ci.low,'hi':ci.high,'method':rr[0]['method'],'p_resolution':rr[0]['p_resolution']})
    family=[]
    for cohort,scenario,group in sorted({(r['cohort'],r['scenario'],r['group']) for r in rows}):
        rr=[r for r in rows if (r['cohort'],r['scenario'],r['group'])==(cohort,scenario,group)]
        subjects={r['subject'] for r in rr};K=len(subjects)
        for rep in sorted({int(r['rep']) for r in rr}):
            x=[r for r in rr if int(r['rep'])==rep]
            if len(x)!=K:raise ValueError('Incomplete profile family')
            family.append({'cohort':cohort,'scenario':scenario,'group':group,'meaning':x[0]['meaning'],'rep':rep,'family_size':K,'any_raw_rejection':int(any(r['p_value']<=alpha for r in x)),'any_bonferroni_rejection':int(any(r['p_value']<=alpha/K for r in x)),'raw_rejections':sum(r['p_value']<=alpha for r in x),'bonferroni_rejections':sum(r['p_value']<=alpha/K for r in x)})
    famsum=[]
    for key in sorted({(r['cohort'],r['scenario'],r['group'],r['meaning']) for r in family}):
        rr=[r for r in family if tuple(r[k] for k in ['cohort','scenario','group','meaning'])==key]
        for m in ['any_raw_rejection','any_bonferroni_rejection']:
            k=sum(r[m] for r in rr);ci=binomtest(k,len(rr)).proportion_ci()
            famsum.append({**dict(zip(['cohort','scenario','group','meaning'],key)),'metric':m,'repetitions':len(rr),'rate':k/len(rr),'lo':ci.low,'hi':ci.high,'interpretation':'FWER only for supported_null; family rejection rate otherwise'})
    csv_write(folder/'recording_calibration_raw.csv',rows);csv_write(folder/'recording_calibration_summary.csv',summary)
    csv_write(folder/'recording_calibration_family_raw.csv',family);csv_write(folder/'recording_calibration_family_summary.csv',famsum)
    return summary

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--stop-after',type=int,default=0);a=p.parse_args(argv)
    cfg=json.loads(Path(a.config).read_text());manifest=Path(cfg['cycle_manifest']);manifest=manifest if manifest.is_absolute() else ROOT/manifest
    profiles=profiles_from_manifest(manifest)
    if cfg.get('expected_profiles') is not None and len(profiles)!=cfg['expected_profiles']:raise ValueError('Profile count differs from frozen config')
    if cfg['repetitions']<1 or cfg['null_permutations']<1 or a.workers<1:raise ValueError('Invalid counts')
    identity={'version':VERSION,'config':cfg,'manifest_sha256':sha(manifest),'sources':{n:sha(ROOT/'revision'/n) for n in ['recording_calibration.py','radar_order.py','storage.py']},'numpy':np.__version__}
    folder=ROOT/cfg['output_root']/('recording_calibration-'+digest(identity)[:16]);print('RUN DIRECTORY',folder,flush=True)
    from . import storage
    storage.CSV_CONTEXT={'run_id':folder.name,'data_mode':'synthetic_recording_calibration'}
    with run_lock(folder):
        (folder/'complete.json').unlink(missing_ok=True);json_write(folder/'run_identity.json',identity);json_write(folder/'profiles.json',profiles)
        json_write(folder/'SCOPE.json',{'generator':'exact cycle-mixture memberships: mu_c,b = w p_segment,b + (1-w) q_cycle; q constant across bins; shared record template and orientation','exact_empirical_quantities':'segment membership, recording grouping, selected B and clean retained cycle counts in each grid','not_reconstructed':'radar memberships, actual phase errors, all real signal dependence or invariant null','dependence':'segments/channels/cycles share recording factors; one group action per recording','full_design_scope':'500 independently generated replicates per declared profile/scenario, exact null or 999 Monte Carlo draws; distinct from legacy independent-cycle pilot','cohorts':'analyzed separately; simulated separately; no pooled biological sample expansion','calibration_limits':'per-profile binomial CIs are descriptive and not corrected across profiles; small discrete groups can be conservative; alternatives do not span weak effects'})
        jobs=[]
        for profile in profiles:
            for start in range(0,cfg['repetitions'],cfg['chunk_size']):
                stop=min(cfg['repetitions'],start+cfg['chunk_size']);key=f"profile/{profile['cohort']}/{profile['subject']}/{start:05d}"
                jobs.append((str(folder),key,profile,cfg,start,stop))
        rows=[];completed=0;new=0;reused=0
        try:
            # At most workers pending chunks during an intentional resume test.
            limit=jobs[:a.stop_after] if a.stop_after else jobs
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                futures=[pool.submit(commit_chunk,*job) for job in limit]
                for fut in as_completed(futures):
                    key,rr,n,r=fut.result();rows.extend(rr);completed+=1;new+=n;reused+=r
                    print('COMMITTED',completed,'/',len(jobs),key,flush=True)
                    json_write(folder/'progress.json',{'committed_chunks_this_invocation':completed,'planned_chunks':len(jobs),'new':new,'reused':reused})
            if a.stop_after and len(limit)<len(jobs):
                json_write(folder/'interrupted.json',{'reason':'controlled stop after chunk commits; same config resumes'});raise SystemExit(75)
            rows.sort(key=lambda r:tuple(r[k] for k in ['cohort','subject','scenario','group','rep']))
            summary=summarize(rows,folder,cfg['alpha'])
            assert len(rows)==len(profiles)*len(cfg['scenarios'])*len(GROUPS)*cfg['repetitions']
            json_write(folder/'complete.json',{'profiles':len(profiles),'cases':len(summary),'rows':len(rows),'repetitions_per_case':cfg['repetitions'],'new_chunks':new,'reused_chunks':reused});print('COMPLETE',len(rows),flush=True)
        except Exception as e:json_write(folder/'last_error.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
