"""Isolated-process full-recording benchmark. Run on the target device.
python -m revision.benchmark --model <outer checkpoint.joblib> --record <csv>
                           --output benchmark.json --repeats 20
"""
from pathlib import Path
import argparse,json,platform,time,subprocess,sys,os
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from pglts.signal.preprocess import highpass,normalise
from pglts.signal.features import FeatureExtractor,PGLTS_CORE
from pglts.signal.segmentation import harmonic_ratio,frame_bounds
from .models import _cached_cycles,get_cycles
from .storage import json_write

def raw_pipeline(model,amplitude,phase):
    t=time.perf_counter();amp=normalise(highpass(amplitude),'std');cpx=normalise(highpass(amplitude*np.exp(1j*phase)),'std');front=time.perf_counter()-t
    extractor=FeatureExtractor(names=PGLTS_CORE);frames=[];t=time.perf_counter()
    for start in frame_bounds(len(amp),2000,1000):
        a=amp[start:start+2000];z=cpx[start:start+2000]
        if harmonic_ratio(a)<=model.options['hr_threshold']:continue
        f={'id':f'benchmark:{start}','amp':a.astype(np.float32),'cpx':z.astype(np.complex64),'window':extractor.transform(a)}
        if model.c['view'] in ('combined','phase_ordered','phase_occ') and len(get_cycles(f,model.c['B'],model.options['max_cycles']))<2:continue
        frames.append(f)
    extraction=time.perf_counter()-t
    if not frames:raise ValueError('Benchmark recording has no accepted segments.')
    t=time.perf_counter();pred=model.predict(frames);inference=time.perf_counter()-t
    return {'front_end_s':front,'framing_window_and_cycle_s':extraction,'descriptor_and_classifier_s':inference,'total_s':front+extraction+inference,'segments':len(frames),'predictions':pred.tolist()}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--record',required=True);p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=20);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if not a.worker:
        subprocess.run([sys.executable,'-m','revision.benchmark','--model',a.model,'--record',a.record,'--output',a.output,'--repeats',str(a.repeats),'--worker'],check=True);return
    model=joblib.load(a.model);threadpool_limits(limits=model.options['threads']);data=np.loadtxt(a.record,delimiter=',',skiprows=1)
    raw_pipeline(model,data[:,0],data[:,1]);rows=[]
    for _ in range(a.repeats):
        _cached_cycles.cache_clear();rows.append(raw_pipeline(model,data[:,0],data[:,1]))
    result={'hardware':platform.platform(),'processor':platform.processor(),'model_bytes':Path(a.model).stat().st_size,'recording_samples':len(data),'acquisition_duration_s':len(data)/1000,'repeats':a.repeats,'threads':model.options['threads'],'energy_joules':None,'scope':'offline full-recording pipeline, identical forward-backward filter context; not 2-second streaming latency','measurements':rows}
    for key in ['front_end_s','framing_window_and_cycle_s','descriptor_and_classifier_s','total_s']:
        result[key+'_median']=float(np.median([r[key] for r in rows]));result[key+'_p95']=float(np.quantile([r[key] for r in rows],.95))
    try:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['process_peak_rss_bytes']=int(peak*(1 if sys.platform=='darwin' else 1024))
    except ImportError:result['process_peak_rss_bytes']=None
    result['memory_scope']='isolated process high-water RSS includes imports, model, input and warmup'
    json_write(a.output,result)
    from .storage import csv_write
    csv_write(Path(a.output).with_suffix('.csv'),[{k:v for k,v in r.items() if k!='predictions'} for r in rows])
if __name__=='__main__':main()
