"""Optional explicit hooks for input reconstruction / model inference replay.
Call only in the actual v140 code after reconstructing its own X; do not invent X here.
These hooks never fit models. Capture timestamps are retrospective, not historical proof.
"""
from pathlib import Path
import json,numpy as np
from export_audit import atomic,array_info,csvwrite,jsonwrite

def save_input(folder,tag,X,samples,components,phase,save_tensor=False):
    """samples: ordered dictionaries with id,subject,y,role,fold; X first axis=samples.
    phase is one of historical_fit_hook or retrospective_reconstruction.
    components names all measured/derived sources, including window features if used.
    """
    if phase not in {'historical_fit_hook','retrospective_reconstruction'}:raise ValueError('Explicit capture phase required')
    if Path(tag).name!=tag:raise ValueError('tag must be a filename component')
    a=np.asarray(X)
    if a.dtype.hasobject:raise ValueError('Object arrays unsupported; capture each numeric component separately')
    if len(a)!=len(samples):raise ValueError('Tensor/sample length mismatch')
    required={'id','subject','y','role','fold'}
    if any(not required<=set(r) for r in samples):raise ValueError('Missing sample identity')
    p=Path(folder)/tag;p.mkdir(parents=True,exist_ok=True)
    csvwrite(p/'samples.csv',[dict(r,tensor_index=i,**array_info(a[i])) for i,r in enumerate(samples)])
    jsonwrite(p/'tensor.json',dict(**array_info(a),components=list(components),capture_phase=phase,axis_0='sample; see samples.csv',claim_scope='captured X only; caller must be the exact run implementation'))
    if save_tensor:
        def w(t):
            with t.open('wb') as f:np.save(f,a,allow_pickle=False)
        atomic(p/'tensor.npy',w)

def save_replay(folder,tag,samples,pred,expected=None):
    """Record inference from an already fitted model, aligned by explicit ID."""
    pred=list(pred)
    if len(samples)!=len(pred):raise ValueError('Replay length mismatch')
    if expected is not None and len(expected)!=len(pred):raise ValueError('Expected prediction length mismatch')
    rows=[dict(r,pred=str(p),**({'expected_pred':str(expected[i]),'match':int(str(p)==str(expected[i]))} if expected is not None else {})) for i,(r,p) in enumerate(zip(samples,pred))]
    csvwrite(Path(folder)/(tag+'_replay.csv'),rows)
