"""Per-recording checkpointed extraction; unambiguous segment IDs, raw-frame support."""
from pathlib import Path
import numpy as np
from pglts.data.polliradar import index
from pglts.signal.preprocess import highpass,normalise
from pglts.signal.features import FeatureExtractor,PGLTS_CORE
from pglts.signal.segmentation import harmonic_ratio,frame_bounds
from .phase import cycles
from .storage import sha

def synthetic_records():
    records=[];rng=np.random.default_rng(123)
    for k in range(3):
        for subject in range(4):
            t=np.arange(4000)/1000;f=85+30*k+rng.normal(0,1)
            amp=1+.1*np.sin(2*np.pi*f*t)+.025*np.sin(4*np.pi*f*t+.3*k)+rng.normal(0,.002,len(t))
            phase=.2*np.sin(2*np.pi*f*t+.2*subject)
            records.append((f'synthetic_{k}#{subject}',f'synthetic_{k}',amp,phase))
    return records

def input_identity(cfg):
    if cfg['data_mode']=='synthetic':return [{'name':'SYNTHETIC_ONLY','sha256':'generator-v1-seed123'}]
    records=index(Path(cfg['data_root']))
    items=[{'name':r.path.name,'sha256':sha(r.path)} for r in records]
    if cfg.get('verify_original_checksums',True):
        checksum_path=Path(__file__).resolve().parents[1]/'dataset'/'SHA256SUMS.txt'
        expected={}
        for line in checksum_path.read_text().splitlines():
            value,name=line.split(maxsplit=1)
            if name.startswith('polliradar/'):expected[Path(name).name]=value
        if set(expected)!={r['name'] for r in items}:raise ValueError('Input recording list differs from original Polliradar checksum manifest.')
        for r in items:
            if r['sha256']!=expected[r['name']]:raise ValueError('Checksum mismatch: '+r['name'])
    return items

def prepare(cfg,store):
    synthetic=cfg['data_mode']=='synthetic'
    records=synthetic_records() if synthetic else index(cfg['data_root'])
    all_frames=[];audit=[];extractor=FeatureExtractor(names=PGLTS_CORE)
    for ri,rec in enumerate(records):
        def build(rec=rec,ri=ri):
            if synthetic:
                subject,species,amplitude,phase=rec;recording=subject.replace('#','_')
            else:
                amplitude,phase=rec.load();subject=rec.subject_key;species=rec.species;recording=rec.path.stem
            amp=normalise(highpass(amplitude),'std');cpx=normalise(highpass(amplitude*np.exp(1j*phase)),'std')
            frames=[];counts={'recording':recording,'subject':subject,'species':species,'total_frames':0,'hr_accepted':0,'cycle_accepted':0}
            for start in frame_bounds(len(amp),2000,1000):
                counts['total_frames']+=1;a=amp[start:start+2000];z=cpx[start:start+2000]
                hr=harmonic_ratio(a)
                if hr<=cfg['hr_threshold']:continue
                counts['hr_accepted']+=1
                cyc=cycles(a,z,8,cfg['max_cycles'])
                if len(cyc)<2:continue
                counts['cycle_accepted']+=1
                frames.append({'id':f'{recording}:{start}:2000','subject':subject,'y':species,'amp':a.astype(np.float32),'cpx':z.astype(np.complex64),'window':extractor.transform(a).astype(np.float32)})
            return frames,counts
        frames,counts=store.task(f'prepare/record_{ri:03}',build);all_frames.extend(frames);audit.append(counts)
    if not all_frames:raise ValueError('No accepted segments. Inspect data and HR threshold.')
    ids=[x['id'] for x in all_frames]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate segment IDs.')
    return all_frames,audit
