"""Nested evaluation and all result exports. No outer-test model selection."""
import itertools,json,time
import numpy as np
from sklearn.metrics import confusion_matrix,f1_score
from .storage import digest,csv_write,json_write
from .protocol import split_subjects,score,summary,paired,subject_rows
from .models import Model

def candidate_space(cfg):
    arms={}
    for arm in cfg['arms']:
        if arm.endswith('_rocket') or arm.endswith('_multirocket') or arm.endswith('_linear'):
            view,method=arm.split('_',1);view='combined' if view=='combined' else 'phase_ordered'
            pool=[{'view':view,'method':method,'B':b,'C':c,'kernels':k,'seed':cfg['seed']} for b,c,k in itertools.product(cfg['bins'],cfg['C_values'],cfg['kernels'] if method!='linear' else [0])]
        elif arm=='static_svm':pool=[{'view':'window','method':'svm','C':c} for c in cfg['C_values']]
        else:
            pool=[{'view':arm,'method':'linguistic','partition':p,'states':a,'B':b,'l1':l,'lg':l/2} for p,a,b,l in itertools.product(cfg['partitions'],cfg['states'],cfg['bins'] if arm in ('phase_occ','phase_ordered','combined') else [8],cfg['penalties'])]
        rng=np.random.default_rng(cfg['seed']);order=rng.permutation(len(pool))[:cfg['budget_per_arm']]
        arms[arm]=[dict(pool[i],candidate_id=digest(pool[i])[:12]) for i in order]
    return arms

def nested(frames,cfg,store,folder):
    folder.mkdir(exist_ok=True,parents=True);labels=sorted({f['y'] for f in frames});spaces=candidate_space(cfg)
    json_write(folder/'search_space.json',spaces)
    outer=split_subjects(frames,np.arange(len(frames)),cfg['outer_folds'],cfg['seed'])
    manifest=[];inner_rows=[];selected_rows=[];predictions=[];results={};metadata=[]
    def predict_rows(model,ix,fold,arm):
        fs=[frames[i] for i in ix];scores=model.scores(fs);pred=model.classes_[scores.argmax(1)]
        return [{'id':f['id'],'subject':f['subject'],'y':f['y'],'pred':str(p),'fold':fold,'arm':arm,**{f'score_{c}':float(v) for c,v in zip(model.classes_,s)}} for f,p,s in zip(fs,pred,scores)]
    for fold,(tr,te) in enumerate(outer):
        inner=split_subjects(frames,tr,cfg['inner_folds'],cfg['seed']+fold+1)
        for level,which,splits in [('outer',fold,[(tr,te)]),('inner',fold,inner)]:
            for j,(a,b) in enumerate(splits):
                for role,ix in [('train',a),('test',b)]:
                    for subject in sorted({frames[i]['subject'] for i in ix}):manifest.append({'level':level,'outer_fold':which,'inner_fold':j if level=='inner' else '', 'role':role,'subject':subject,'species':next(frames[i]['y'] for i in ix if frames[i]['subject']==subject)})
        best_for_selection=[]
        for arm,space in spaces.items():
            scores=[]
            for candidate in space:
                # Pool all inner-held-out predictions before fixed-label macro-F1.
                ys=[];ps=[];fit_seconds=0.;converged=True
                for infold,(it,iv) in enumerate(inner):
                    def evaluate(it=it,iv=iv,c=candidate):
                        t=time.perf_counter();m=Model(c,cfg).fit([frames[i] for i in it]);pred=m.predict([frames[i] for i in iv])
                        return {'y':[frames[i]['y'] for i in iv],'pred':pred,'seconds':time.perf_counter()-t,'converged':m.converged}
                    r=store.task(f'inner/{fold}/{arm}/{candidate["candidate_id"]}/{infold}',evaluate)
                    ys.extend(r['y']);ps.extend(r['pred']);fit_seconds+=r['seconds'];converged &= r['converged']
                value=float(f1_score(ys,ps,labels=labels,average='macro',zero_division=0))
                record={'outer_fold':fold,'arm':arm,'candidate_id':candidate['candidate_id'],'inner_macro_f1':value,'fit_seconds':fit_seconds,'converged':converged,'config':json.dumps(candidate,sort_keys=True)}
                inner_rows.append(record);scores.append((value,candidate['candidate_id'],candidate))
            # Deterministic score then ID tie-break; no test metric read here.
            value,_,best=sorted(scores,key=lambda r:(-r[0],r[1]))[0]
            def fit_outer(best=best):return Model(best,cfg).fit([frames[i] for i in tr])
            model=store.task(f'outer/{fold}/{arm}',fit_outer)
            rows=predict_rows(model,te,fold,arm);predictions.extend(rows)
            semantics={}
            for part in ['win_sem','cyc_sem']:
                if hasattr(model,part):
                    sem=getattr(model,part);semantics[part]={key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in vars(sem).items() if isinstance(value,(np.ndarray,str,int,float,tuple,list))}
            metadata.append({'fold':fold,'arm':arm,'candidate':best,'descriptor_names':model.names,'semantic_state':semantics,'checkpoint':f'checkpoints/outer__{fold}__{arm}.joblib'})
            results[(fold,arm)]={'model':model,'train':tr,'test':te,'rows':rows,'candidate':best}
            selected_rows.append({'outer_fold':fold,'arm':arm,'inner_macro_f1':value,'config':json.dumps(best,sort_keys=True),'input_features':len(model.names),'active_features':model.active(),'converged':model.converged})
            if best['method']=='linguistic':best_for_selection.append((value,best['candidate_id'],arm))
        if best_for_selection:
            _,_,chosen=sorted(best_for_selection,key=lambda r:(-r[0],r[1]))[0]
            rows=[dict(r,arm='selected_linguistic') for r in results[(fold,chosen)]['rows']];predictions.extend(rows)
            selected_rows.append({'outer_fold':fold,'arm':'selected_linguistic','source_arm':chosen,'config':json.dumps(results[(fold,chosen)]['candidate'],sort_keys=True)})
        csv_write(folder/'inner_search.csv',inner_rows);csv_write(folder/'selected_configs.csv',selected_rows);csv_write(folder/'oof_predictions.csv',predictions);csv_write(folder/'splits.csv',manifest)
    json_write(folder/'model_metadata.json',metadata)
    json_write(folder/'nested_protocol.json',{'folds':len(outer),'arms':list(spaces),'selection_uses':'inner OOF macro-F1 only'})
    return results,predictions

def analyze(predictions,cfg,folder):
    arms=sorted({r['arm'] for r in predictions});metrics=[];sub=[];conf=[];influence=[]
    for arm in arms:
        rows=[r for r in predictions if r['arm']==arm]
        metrics.extend(dict(r,arm=arm) for r in summary(rows,cfg['bootstrap'],cfg['seed']))
        sr=subject_rows(rows);sub.extend(dict(r,arm=arm) for r in sr)
        labels=sorted({r['y'] for r in rows})
        for level,data in [('segment',rows),('subject',sr)]:
            cm=confusion_matrix([r['y'] for r in data],[r['pred'] for r in data],labels=labels)
            for i,a in enumerate(labels):
                for j,b in enumerate(labels):conf.append({'arm':arm,'level':level,'true':a,'predicted':b,'count':int(cm[i,j])})
        point=f1_score([r['y'] for r in rows],[r['pred'] for r in rows],labels=labels,average='macro',zero_division=0)
        for s in sorted({r['subject'] for r in rows}):
            kept=[r for r in rows if r['subject']!=s]
            v=f1_score([r['y'] for r in kept],[r['pred'] for r in kept],labels=labels,average='macro',zero_division=0)
            influence.append({'arm':arm,'omitted_subject':s,'delta_vs_all':v-point,'scope':'fixed OOF predictions; not LOSO refit'})
    deltas=[]
    for a,b in [('combined','occupancy'),('phase_ordered','phase_occ'),('phase_ordered','phase_rocket'),('combined','combined_rocket'),('phase_ordered','phase_linear'),('combined','combined_linear')]:
        if a in arms and b in arms:
            for level in ['segment','subject']:
                deltas.append({'a':a,'b':b,**paired([r for r in predictions if r['arm']==a],[r for r in predictions if r['arm']==b],cfg['bootstrap'],cfg['seed'],level)})
    for name,rows in [('metrics_with_ci',metrics),('subject_predictions',sub),('confusion_matrices',conf),('subject_influence',influence),('paired_deltas',deltas)]:csv_write(folder/(name+'.csv'),rows)
