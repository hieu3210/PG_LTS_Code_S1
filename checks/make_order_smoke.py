"""Build SYNTHETIC phase-only checkpoint fixtures. Never a radar study result.
The order analysis consumes the fitted membership map, not classifier coefficients.
"""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from revision import ROOT
from revision.storage import Store,json_write,csv_write
from revision.models import Model,get_cycles
from pglts.linguistic.semantize import make_semantizer

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='validation/order_smoke_source');a=p.parse_args();out=ROOT/a.output;store=Store(out/'checkpoints');manifest=[];frames=[];phase=[];splits=[];selected=[]
 cfg={'data_mode':'synthetic','name':'SYNTHETIC_ORDER_FIXTURE','max_cycles':8,'threads':1,'seed':20261101}
 identity={'config':cfg,'sources':json.loads((ROOT/'LEGACY_COMPATIBILITY.json').read_text())['sources'],'scope':'synthetic fixture; no real-data or classifier performance claims'};json_write(out/'run_identity.json',identity)
 for species in range(2):
  for individual in range(2):
   subject=f'synthetic_{species}#{individual}';rec=f'synthetic_{species}_{individual}';t=np.arange(4000)/1000;freq=90+30*species
   amp=np.sin(2*np.pi*freq*t)+.2*np.sin(4*np.pi*freq*t+.3*individual);z=(1+.1*np.sin(2*np.pi*freq*t))*np.exp(.2j*np.sin(2*np.pi*freq*t+.2))
   fs=[{'id':f'{rec}:{start}:2000','subject':subject,'y':f'synthetic_{species}','amp':amp[start:start+2000].astype(np.float32),'cpx':z[start:start+2000].astype(np.complex64),'window':np.zeros((19,10),np.float32)} for start in [0,1000,2000]]
   store.task(f'prepare/record_{species*2+individual:03}',lambda fs=fs:(fs,{'scope':'synthetic'}));frames+=fs
   manifest.extend({'id':f['id'],'subject':subject,'species':f['y']} for f in fs);phase.extend({'id':f['id'],'subject':subject,'y':f['y'],'pred':f['y'],'fold':individual,'arm':'phase_ordered'} for f in fs)
 subjects=sorted({f['subject'] for f in frames})
 for fold in range(2):
  train=[f for f in frames if int(f['subject'].split('#')[1])!=fold]
  c={'view':'phase_ordered','method':'linguistic','B':8,'states':4,'partition':'ha_soft','l1':.1,'lg':.05};m=Model(c,cfg)
  m.cyc_sem=make_semantizer('ha_soft',n_states=4).fit(np.concatenate([get_cycles(f,8,8) for f in train]));store.task(f'outer/{fold}/phase_ordered',lambda m=m:m)
  selected.append({'outer_fold':fold,'arm':'phase_ordered','config':json.dumps(c)})
  splits.extend({'level':'outer','outer_fold':fold,'role':'test' if int(sub.split('#')[1])==fold else 'train','subject':sub} for sub in subjects)
 for name,rows in [('segment_manifest',manifest),('oof_predictions',phase),('splits',splits),('selected_configs',selected)]:csv_write(out/(name+'.csv'),rows)
 print('WROTE SYNTHETIC FIXTURE',out)
if __name__=='__main__':main()
