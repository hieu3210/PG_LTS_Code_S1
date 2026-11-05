"""Per-class retained-score rules. Never equates aggregate accuracy with fidelity."""
import itertools
import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score
from .storage import csv_write
from .protocol import clustered_draw,SubjectBootstrap
from .models import Model

def retained_scores(z,coef,intercept,importance,k):
    mask=np.zeros_like(coef,bool)
    for j in range(len(coef)):
        active=np.flatnonzero(np.abs(coef[j])>1e-10)
        chosen=active[np.argsort(-importance[j,active],kind='stable')[:k]];mask[j,chosen]=True
    return z@(coef*mask).T+intercept,mask

def run(frames,results,cfg,store,folder):
    metrics=[];rules=[];preds=[];sets={};comparisons=[];frontier=[]
    expanded=dict(results)
    for (fold,arm),entry in list(results.items()):
        if arm not in ('full_window','combined'):continue
        reference_candidate=dict(entry['candidate'],l1=min(cfg['penalties']),lg=min(cfg['penalties'])/2)
        reference=store.task(f'sparse_reference/{fold}/{arm}',lambda:Model(reference_candidate,cfg).fit([frames[i] for i in entry['train']]))
        test=[frames[i] for i in entry['test']];reference_scores=reference.scores(test);reference_pred=reference.classes_[reference_scores.argmax(1)]
        for penalty in cfg['penalties']:
            c=dict(entry['candidate'],l1=penalty,lg=penalty/2)
            sparse=store.task(f'sparse_frontier/{fold}/{arm}/{penalty}',lambda c=c:Model(c,cfg).fit([frames[i] for i in entry['train']]))
            value=sparse.scores(test);delta=(value-value.mean(1,keepdims=True))-(reference_scores-reference_scores.mean(1,keepdims=True));p=sparse.classes_[value.argmax(1)]
            frontier.append({'fold':fold,'arm':arm,'penalty':penalty,'active_features':sparse.active(),'macro_f1':float(f1_score([f['y'] for f in test],p,labels=sparse.classes_,average='macro',zero_division=0)),'agreement_with_weak_reference':float(np.mean(p==reference_pred)),'centered_score_rmse_vs_reference':float(np.sqrt(np.mean(delta**2))),'selection_scope':'prespecified penalty frontier; do not choose penalty from outer-test scores'})
            expanded[(fold,arm+f':penalty={penalty}')]=dict(entry,model=sparse)
    results=expanded
    for (fold,arm),entry in results.items():
        if entry['candidate']['method']!='linguistic':continue
        model=entry['model'];test=[frames[i] for i in entry['test']];z=model.transform(test);coef=model.clf.coef_;base=model.clf.decision_function(z);importance=np.abs(coef)*model.train_abs
        for k in cfg['rule_k']:
            def build(k=k):
                only,mask=retained_scores(z,coef,model.clf.intercept_,importance,k)
                centered=base-base.mean(1,keepdims=True);delta=(only-only.mean(1,keepdims=True))-centered
                y=np.array([f['y'] for f in test]);full=model.classes_[base.argmax(1)];p=model.classes_[only.argmax(1)]
                return {'agreement':float(np.mean(p==full)),'macro_f1':float(f1_score(y,p,labels=model.classes_,average='macro',zero_division=0)),'centered_score_rmse':float(np.sqrt(np.mean(delta**2))),'probability_mae':float(np.abs(softmax(only,1)-softmax(base,1)).mean()),'mask':mask,'pred':p,'full':full}
            r=store.task(f'fidelity/{fold}/{arm}/top{k}',build)
            metrics.append({'fold':fold,'arm':arm,'top_k_per_class':k,**{a:b for a,b in r.items() if a not in ('mask','pred','full')},'selected_coefficients':int(r['mask'].sum())})
            key=(arm,k);identity=set()
            for j,i in zip(*np.where(r['mask'])):
                sign='+' if coef[j,i]>0 else '-';name=model.names[i];cls=str(model.classes_[j]);identity.add((cls,name,sign))
                rules.append({'fold':fold,'arm':arm,'top_k_per_class':k,'class':cls,'descriptor':name,'sign':sign,'weight':float(coef[j,i]),'training_mean_abs_contribution':float(importance[j,i])})
            sets.setdefault(key,[]).append(identity)
            preds.extend({'id':f['id'],'subject':f['subject'],'y':f['y'],'pred':str(p),'full_pred':str(q),'arm':arm,'fold':fold,'top_k_per_class':k} for f,p,q in zip(test,r['pred'],r['full']))
    stability=[]
    for (arm,k),ss in sets.items():
        js=[len(a&b)/len(a|b) if a|b else 1. for a,b in itertools.combinations(ss,2)]
        stability.append({'arm':arm,'top_k_per_class':k,'mean_pairwise_jaccard':float(np.mean(js)) if js else '', 'folds':len(ss),'scope':'same textual antecedent/class/sign; numeric learned thresholds may differ'})
    # CI of prediction agreement clusters whole insects, independently of accuracy.
    agreement_ci=[]
    for arm,k in sets:
        rows=[r for r in preds if r['arm']==arm and r['top_k_per_class']==k];rng=np.random.default_rng(cfg['seed']);match=np.array([r['pred']==r['full_pred'] for r in rows]);sampler=SubjectBootstrap(rows);draws=[float(match[sampler.draw(rng)].mean()) for _ in range(cfg['bootstrap'])];lo,hi=np.quantile(draws,[.025,.975]);agreement_ci.append({'arm':arm,'top_k_per_class':k,'agreement':float(match.mean()),'lo':lo,'hi':hi})
    # Honest comparison between separately inner-selected full-window/combined models.
    for fold in sorted({k[0] for k in results}):
        if (fold,'combined') in results and (fold,'full_window') in results:
            a=results[(fold,'combined')]['rows'];b=results[(fold,'full_window')]['rows'];lookup={r['id']:r for r in b}
            comparisons.append({'fold':fold,'comparison':'combined_vs_full_window','prediction_agreement':float(np.mean([r['pred']==lookup[r['id']]['pred'] for r in a])),'scope':'different inner-selected configurations; not a sparse approximation guarantee'})
    for name,rows in [('rule_fidelity',metrics),('rule_agreement_ci',agreement_ci),('rules_by_fold',rules),('rule_predictions',preds),('rule_stability',stability),('model_prediction_agreement',comparisons),('sparsity_frontier',frontier)]:csv_write(folder/(name+'.csv'),rows)
