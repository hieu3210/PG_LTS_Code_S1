import json,os
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp
from revision.storage import Store,run_lock
from revision.solver import StableSparseGroup
from revision.protocol import split_subjects,clustered_draw,paired
from revision.order import residual,bh,randomize,statistic
from revision.models import permute_block,Model
from revision.data import prepare
from revision.experiments import candidate_space
from revision.fidelity import retained_scores


def test_solver_unpenalized_matches_independent_optimizer():
    rng=np.random.default_rng(4);X=rng.normal(size=(40,3));y=np.r_[np.zeros(25,int),np.ones(10,int),np.full(5,2)]
    model=StableSparseGroup(0,0,5000,1e-8).fit(X,y,['g']*3)
    weights=40/(3*np.bincount(y))[y]
    def fun(v):
        W=v[:9].reshape(3,3);b=v[9:];z=X@W+b
        return np.mean(weights*(logsumexp(z,axis=1)-z[np.arange(40),y]))
    r=minimize(fun,np.zeros(12),method='BFGS',options={'gtol':1e-9})
    assert abs(model.objective_-r.fun)<2e-6
    assert model.n_iter_<=5000 and np.isfinite(model.objective_)


def test_atomic_restart_and_corruption(tmp_path):
    calls=[];s=Store(tmp_path,stop_after=1)
    with pytest.raises(InterruptedError):s.task('a',lambda:calls.append(1) or {'a':1})
    s=Store(tmp_path);assert s.task('a',lambda:None)=={'a':1};assert s.reused==1
    (tmp_path/'a.joblib').write_bytes(b'corrupt')
    assert s.task('a',lambda:{'a':2})=={'a':2}
    assert len(calls)==1


def test_lock_excludes_concurrent_process(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with run_lock(tmp_path):pass


def test_splits_keep_subjects_and_all_train_classes():
    f=[{'subject':f'{c}:{s}','y':str(c)} for c in range(3) for s in range(4) for _ in range(2)]
    for tr,te in split_subjects(f,np.arange(len(f)),4,1):
        assert not {f[i]['subject'] for i in tr}&{f[i]['subject'] for i in te}
        assert len({f[i]['y'] for i in tr})==3
        for a,b in split_subjects(f,tr,3,2):assert set(a)|set(b)==set(tr)


def test_permutation_preserves_each_cycle_and_shared_channels():
    x=np.arange(2*8*3).reshape(2,8,3);p=permute_block(x,3,'s')
    for a,b in zip(x,p):assert np.array_equal(np.sort(a,axis=0),np.sort(b,axis=0))
    assert np.all(p[:,:,1]-p[:,:,0]==1)


def test_permutation_expectation_exact_enumeration():
    import itertools
    mu=np.array([[[[1.,0.]],[[.2,.8]],[[0.,1.]]]])
    residuals=[residual(mu[:,order]) for order in itertools.permutations(range(3))]
    assert np.max(np.abs(np.mean(residuals,axis=0)))<1e-14
    assert any(np.linalg.norm(v)>0 for v in residuals)


def test_paired_refuses_different_cohorts():
    a=[{'id':'1','subject':'s','y':'a','pred':'a'}];b=[dict(a[0],id='2')]
    with pytest.raises(ValueError):paired(a,b,10,1)


def test_rule_scores_all_active_reproduce_full():
    rng=np.random.default_rng(3);z=rng.normal(size=(10,4));W=rng.normal(size=(3,4));b=rng.normal(size=3)
    s,mask=retained_scores(z,W,b,np.abs(W),100)
    assert np.allclose(s,z@W.T+b) and mask.all()


def test_fit_does_not_read_test_labels(tmp_path):
    root=Path(__file__).resolve().parents[1];cfg=json.loads((root/'configs/smoke.json').read_text());frames,_=prepare(cfg,Store(tmp_path))
    train=[f for f in frames if f['subject'].endswith(('#0','#1'))];test=[f for f in frames if f['subject'].endswith(('#2','#3'))]
    c=candidate_space(cfg)['combined'][0];model=Model(c,cfg).fit(train);a=model.scores(test)
    changed=[dict(f,y='unseen_test_label') for f in test];b=model.scores(changed)
    assert np.array_equal(a,b)


def test_bh_adjustment():assert np.allclose(bh([.01,.04,.03]),[.03,.04,.04])


def test_cli_interrupt_resume_and_no_refit_on_third_run(tmp_path):
    import subprocess,sys
    root=Path(__file__).resolve().parents[1];cfg=json.loads((root/'configs/smoke.json').read_text())
    cfg.update(name='integration_resume',output_root=str(tmp_path/'runs'),arms=['occupancy','combined'])
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    command=[sys.executable,'-m','revision.cli','--config',str(path),'--stage','nested']
    r=subprocess.run(command+['--stop-after','3'],cwd=root,capture_output=True,text=True)
    assert r.returncode==75,r.stderr
    r=subprocess.run(command,cwd=root,capture_output=True,text=True);assert r.returncode==0,r.stderr
    run=next((tmp_path/'runs').iterdir());first=(run/'oof_predictions.csv').read_bytes()
    assert json.loads((run/'nested_protocol.json').read_text())['folds']==2
    r=subprocess.run(command,cwd=root,capture_output=True,text=True);assert r.returncode==0,r.stderr
    assert first==(run/'oof_predictions.csv').read_bytes()
    marker=json.loads((run/'nested_protocol.json').read_text());assert marker['arms']==['occupancy','combined']
    # The final stage receipt records zero newly fitted/prepared checkpoint tasks.
    assert json.loads((run/'nested_complete.json').read_text())['new_tasks']==0


def test_multirocket_actual_short_cycle_backend(tmp_path):
    root=Path(__file__).resolve().parents[1];cfg=json.loads((root/'configs/smoke.json').read_text());frames,_=prepare(cfg,Store(tmp_path))
    c={'view':'combined','method':'multirocket','B':4,'C':1,'kernels':168,'seed':1}
    train=[f for f in frames if f['subject'].endswith(('#0','#1'))];test=[f for f in frames if f['subject'].endswith(('#2','#3'))]
    m=Model(c,cfg).fit(train);s=m.scores(test)
    assert s.shape==(len(test),3) and np.isfinite(s).all()


def test_process_kill_releases_lock_and_keeps_completed_tasks(tmp_path):
    import subprocess,sys,time
    root=Path(__file__).resolve().parents[1]
    code='''import sys,time
from revision.storage import Store,run_lock
from pathlib import Path
folder=Path(sys.argv[1])
with run_lock(folder):
 s=Store(folder/'checkpoints')
 s.task('completed',lambda:17)
 s.task('interrupted',lambda:time.sleep(60))
'''
    p=subprocess.Popen([sys.executable,'-c',code,str(tmp_path)],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+10
        while not (tmp_path/'checkpoints/completed.done.json').exists():
            assert time.monotonic()<deadline and p.poll() is None
            time.sleep(.02)
        p.kill();p.wait(timeout=5)
        with run_lock(tmp_path):
            s=Store(tmp_path/'checkpoints');assert s.task('completed',lambda:0)==17
            assert s.task('interrupted',lambda:23)==23
    finally:
        if p.poll() is None:p.kill();p.wait()


def test_full_and_combined_affine_reparameterization():
    import runpy
    root=Path(__file__).resolve().parents[1]
    r=runpy.run_path(str(root/'checks/verify_affine_reduction.py'))
    assert r['full']['reduced_coordinates']==225
    assert r['combined']['reduced_coordinates']==84


def test_precomputed_bootstrap_preserves_duplicate_subject_draws():
    from revision.protocol import SubjectBootstrap
    rows=[{'subject':f'{c}:{s}','y':str(c)} for c in range(2) for s in range(3) for _ in range(s+1)]
    a=np.random.default_rng(9);b=np.random.default_rng(9)
    old=[]
    for c in sorted({r['y'] for r in rows}):
        subjects=sorted({r['subject'] for r in rows if r['y']==c})
        for subject in a.choice(subjects,len(subjects),replace=True):old.extend(i for i,r in enumerate(rows) if r['subject']==subject)
    assert np.array_equal(old,SubjectBootstrap(rows).draw(b))


def test_quantile_two_state_four_bin_shapes():
    root=Path(__file__).resolve().parents[1];cfg=json.loads((root/'configs/smoke.json').read_text());rng=np.random.default_rng(6);frames=[]
    for cls in range(3):
        for j in range(2):
            t=np.arange(2000)/1000;f=90+cls*20;amp=np.sin(2*np.pi*f*t)+.1*np.sin(4*np.pi*f*t+j*.1);z=np.exp(1j*.3*np.sin(2*np.pi*f*t))
            frames.append({'id':f'{cls}:{j}','subject':f'{cls}:{j}','y':str(cls),'amp':amp.astype(np.float32),'cpx':z.astype(np.complex64),'window':rng.normal(size=(19,10))})
    for view in ['occupancy','phase_ordered','combined']:
        c={'view':view,'method':'linguistic','partition':'quantile','states':2,'B':4,'l1':.01,'lg':.005}
        model=Model(c,cfg).fit(frames);scores=model.scores(frames)
        assert scores.shape==(6,3) and np.isfinite(scores).all()
