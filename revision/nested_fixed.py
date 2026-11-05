"""Additive copy of legacy nested fitting with frozen candidate grids and explicit split seed.
No legacy source edited. Actual fits and train-only transformations are unchanged.
"""
import json,time
import numpy as np
from sklearn.metrics import f1_score
from .storage import csv_write,json_write
from .protocol import split_subjects
from .models import Model

def nested_fixed(frames,cfg,store,folder,spaces,split_seed):
    folder.mkdir(exist_ok=True,parents=True);labels=sorted({f['y'] for f in frames});spaces=spaces
    json_write(folder/'search_space.json',spaces)
    outer=split_subjects(frames,np.arange(len(frames)),cfg['outer_folds'],split_seed)
    manifest=[];inner_rows=[];selected_rows=[];predictions=[];results={};metadata=[]
    def predict_rows(model,ix,fold,arm):
        fs=[frames[i] for i in ix];scores=model.scores(fs);pred=model.classes_[scores.argmax(1)]
        return [{'id':f['id'],'subject':f['subject'],'y':f['y'],'pred':str(p),'fold':fold,'arm':arm,**{f'score_{c}':float(v) for c,v in zip(model.classes_,s)}} for f,p,s in zip(fs,pred,scores)]
    for fold,(tr,te) in enumerate(outer):
        inner=split_subjects(frames,tr,cfg['inner_folds'],split_seed+fold+1)
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
