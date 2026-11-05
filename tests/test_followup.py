import csv,itertools,json,subprocess,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from revision.followup_calibration import memberships,subject_stat,batch_stats,draws,single
from revision.order import statistic
from revision.followup import summarize,read_csv
from revision.storage import Store
from revision import ROOT


def test_aggregated_statistic_matches_legacy_and_ragged_reference():
    rng=np.random.default_rng(9);mu=memberships(rng.uniform(size=(3,5,8,3)))
    ref=np.mean([statistic(x) for x in mu]);assert abs(subject_stat(mu)-ref)<1e-14
    assert abs(batch_stats(mu[None])[0]-ref)<1e-14
    counts=[2,5,3];ref=np.mean([statistic(x[:n]) for x,n in zip(mu,counts)])
    assert abs(subject_stat(mu,counts)-ref)<1e-14
    assert abs(batch_stats(mu[None],counts)[0]-ref)<1e-14


def test_shared_cycle_randomization_keeps_overlap_and_channels():
    rng=np.random.default_rng(1);mu=memberships(rng.uniform(size=(5,4,3)));ids=np.array([[0,1,2],[2,3,4]])
    for group in ['uniform_bins','cyclic_shift']:
        x=draws(mu,ids,group,'coherent_cycle_id',rng,4)
        assert np.array_equal(x[:,0,2],x[:,1,0])
        for row in x[:,0,0]:assert sorted(map(tuple,row.reshape(4,-1)))==sorted(map(tuple,mu[0].reshape(4,-1)))


def test_exact_randomization_is_conservative_on_small_finite_group():
    mu=memberships(np.array([.05,.4,.95])[None,:,None])
    vals=np.array([statistic(mu[:,ix]) for ix in itertools.permutations(range(3))])
    ps=np.array([np.mean(vals>=v-1e-12) for v in vals])
    for alpha in [.05,.2,.5]:assert np.mean(ps<=alpha)<=alpha+1e-12


def test_calibration_seed_reproducible_and_mismatch_not_called_fpr():
    c={'seed':7,'channels':3,'states':4,'null_permutations':19,'draw_batch':3,'alpha':.05}
    case={'segments':2,'cycles':4,'bins':4,'generator':'smooth','strength':0,'overlap':False,'group':'uniform_bins','coupling':'independent_occurrence'}
    a=single(c,case,0);assert a==single(c,case,0) and a['p_value']>=1/20
    assert a['meaning']=='rejection_rate_under_null_mismatch'
    case.update(group='cyclic_shift');assert single(c,case,0)['exchangeability_supported_by_generator']


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def test_summary_pools_predictions_and_weights_score_errors(tmp_path):
    src=tmp_path/'source';src.mkdir();out=tmp_path/'out';out.mkdir()
    identity={'sources':json.loads((ROOT/'LEGACY_COMPATIBILITY.json').read_text())['sources'],'config':{'name':'fixture'}}
    (src/'run_identity.json').write_text(json.dumps(identity));truth=['a','b','b','b','a','a'];pred=['a']*6;rows=[]
    for j,(y,p) in enumerate(zip(truth,pred)):rows.append({'id':str(j),'subject':str(j),'y':y,'pred':p,'full_pred':y,'arm':'fixture','fold':0 if j<2 else 1,'top_k_per_class':10})
    write_csv(src/'rule_predictions.csv',rows);write_csv(src/'oof_predictions.csv',rows)
    write_csv(src/'rule_fidelity.csv',[{'arm':'fixture','fold':0,'top_k_per_class':10,'centered_score_rmse':1,'probability_mae':.1},{'arm':'fixture','fold':1,'top_k_per_class':10,'centered_score_rmse':3,'probability_mae':.3}])
    summarize(src,out,Store(out/'ckpt'),5,1)
    r=read_csv(out/'rule_metrics_pooled.csv')[0];assert abs(float(r['pooled_rule_macro_f1'])-f1_score(truth,pred,average='macro'))<1e-14
    r=read_csv(out/'rule_score_fidelity_pooled.csv')[0];assert abs(float(r['centered_score_rmse'])-np.sqrt(38/6))<1e-14


def test_raw_transform_shapes_and_real_backend():
    from revision.raw_baselines import raw_tensor,RawModel
    rng=np.random.default_rng(7);frames=[]
    for i in range(8):
        t=np.arange(2000)/1000;z=np.exp(2j*np.pi*(90+20*(i%2))*t)+.1*rng.normal(size=2000);frames.append({'cpx':z.astype(np.complex64),'y':str(i%2)})
    assert raw_tensor(frames,'raw_iq').shape==(8,2,2000)
    assert raw_tensor(frames,'spectrogram').shape[1:]==(128,59)
    for view,method in [('raw_iq','rocket'),('spectrogram','multirocket')]:
        cfg={'threads':1,'transform_batch':3,'max_iter':3000,'tol':1e-5,'require_convergence':True};c={'view':view,'method':method,'C':1,'kernels':84,'seed':1}
        m=RawModel(c,cfg).fit(frames[:6]);assert len(m.predict(frames[6:]))==2


def test_followup_cli_stop_resume_and_corrupt_checkpoint(tmp_path):
    cfg=json.loads((ROOT/'configs/followup_smoke.json').read_text());cfg['profiles']=cfg['profiles'][:1];cfg['bins']=[4];cfg['scenarios']=cfg['scenarios'][:1];cfg['groups']=['uniform_bins'];cfg['repetitions']=2
    path=tmp_path/'cfg.json';path.write_text(json.dumps(cfg));out=tmp_path/'runs';cmd=[sys.executable,'-m','revision.followup','--stage','calibration','--config',str(path),'--output-root',str(out)]
    r=subprocess.run(cmd+['--stop-after','1'],capture_output=True,text=True);assert r.returncode==75,r.stderr
    r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
    root=next(out.iterdir());r=json.loads((root/'complete.json').read_text());assert r['reused_tasks']==1 and r['new_tasks']==1
    next((root/'checkpoints').glob('*.joblib')).write_bytes(b'broken')
    r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
    r=json.loads((root/'complete.json').read_text());assert r['new_tasks']==1 and r['reused_tasks']==1


def test_cached_expectation_batch_matches_scalar_for_both_groups():
    from revision.followup_calibration import expected_mean
    rng=np.random.default_rng(33);mu=memberships(rng.random((9,8,3)));ids=np.array([[0,1,2],[3,4,5],[6,7,8]])
    counts=[2,3,1];E=expected_mean(mu[ids],counts)
    for group in ['uniform_bins','cyclic_shift']:
        x=draws(mu,ids,group,'independent_occurrence',rng,5)
        actual=batch_stats(x,counts,E)
        expected=np.array([subject_stat(row,counts) for row in x])
        assert np.allclose(actual,expected,atol=1e-14,rtol=0)
