"""Verify post-fit score reparameterization. Requires NumPy; no radar data.
This checks algebra on synthetic simplex trajectories, not classification.
Run from any directory: python checks/verify_affine_reduction.py
"""
from pathlib import Path
import csv
import numpy as np

rng = np.random.default_rng(20261030)
N, L, R, A, K = 200, 19, 10, 4, 5
pairs = [(0, 1), (0, 7), (1, 9), (1, 8), (2, 3)]

def blocks(mu):
    O = mu.mean(1)
    M = np.einsum('nlra,nlrb->nrab', mu[:, :-1], mu[:, 1:])/(mu.shape[1]-1)
    P = np.diagonal(M, axis1=-2, axis2=-1)
    return O, P, M

def reduce_scores(O, P, M, C, pair_list):
    n, r, a = O.shape
    e = C.shape[1]
    raw = np.concatenate([O.reshape(n,-1), P.reshape(n,-1), M.reshape(n,-1), C.reshape(n,-1)],1)
    # A fitted classifier expressed in standardized input coordinates.
    mean = rng.normal(size=raw.shape[1]); scale = rng.uniform(.1, 2, size=raw.shape[1])
    weights_std = rng.normal(size=(K,raw.shape[1])); intercept_std = rng.normal(size=K)
    scores = (raw-mean)/scale @ weights_std.T + intercept_std
    w = weights_std/scale; intercept = intercept_std - (weights_std*mean/scale).sum(1)
    split = np.cumsum([r*a,r*a,r*a*a])
    wo, wp, wm, wc = np.split(w, split, axis=1)
    wo = wo.reshape(K,r,a).copy(); wm=wm.reshape(K,r,a,a).copy(); wc=wc.reshape(K,e,a,a)
    for t in range(a): wm[:,:,t,t] += wp.reshape(K,r,a)[:,:,t]
    # Eliminate co-occurrence final row/column using retained occupancies.
    v_reduced=[]
    for j,(u,v) in enumerate(pair_list):
        z=wc[:,j]
        intercept += z[:,-1,-1]
        wo[:,u,:-1] += z[:,:-1,-1]-z[:,-1,-1,None]
        wo[:,v,:-1] += z[:,-1,:-1]-z[:,-1,-1,None]
        v_reduced.append(z[:,:-1,:-1]-z[:,:-1,-1,None]-z[:,-1,None,:-1]+z[:,-1,-1,None,None])
    intercept += wo[:,:,-1].sum(1)
    wo_red=wo[:,:,:-1]-wo[:,:,-1,None]
    wm=wm.reshape(K,r,a*a)
    intercept += wm[:,:,-1].sum(1)
    wm_red=wm[:,:,:-1]-wm[:,:,-1,None]
    coords=[O[:,:,:-1].reshape(n,-1),M.reshape(n,r,a*a)[:,:,:-1].reshape(n,-1)]
    coeff=[wo_red.reshape(K,-1),wm_red.reshape(K,-1)]
    if e:
        coords.append(C[:,:,:-1,:-1].reshape(n,-1));coeff.append(np.stack(v_reduced,1).reshape(K,-1))
    reduced=np.concatenate(coords,1); wr=np.concatenate(coeff,1)
    new_scores=reduced@wr.T+intercept
    delta=float(np.max(np.abs(scores-new_scores)))
    agreement=float(np.mean(scores.argmax(1)==new_scores.argmax(1)))
    assert delta<1e-10 and agreement==1.0, (delta,agreement)
    return {'raw_coordinates':raw.shape[1], 'reduced_coordinates':reduced.shape[1], 'max_abs_score_error':delta,'prediction_agreement':agreement}

mu=rng.dirichlet(np.ones(A),size=(N,L,R))
O,P,M=blocks(mu)
C=np.stack([np.einsum('nla,nlb->nab',mu[:,:,u],mu[:,:,v])/L for u,v in pairs],1)
full=reduce_scores(O,P,M,C,pairs); assert full['raw_coordinates']==320 and full['reduced_coordinates']==225
# Combined window occupancy plus three cycle-resolved channels.
mu_cycle=rng.dirichlet(np.ones(A),size=(N,8,3));Oc,Pc,Mc=blocks(mu_cycle)
# Reuse general layout with zero P/M on the 10 occupancy-only channels excluded
# by embedding their coefficients as zero and checking an explicit compact score.
raw=np.concatenate([O.reshape(N,-1),Oc.reshape(N,-1),Pc.reshape(N,-1),Mc.reshape(N,-1)],1)
mean=rng.normal(size=112);scale=rng.uniform(.1,2,size=112)
wz=rng.normal(size=(K,112));bz=rng.normal(size=K)
original=(raw-mean)/scale@wz.T+bz
w=wz/scale;b=bz-(wz*mean/scale).sum(1)
wo,woc,wpc,wmc=np.split(w,[40,52,64],axis=1)
wo=wo.reshape(K,10,4);woc=woc.reshape(K,3,4);wmc=wmc.reshape(K,3,4,4).copy()
for j in range(4):wmc[:,:,j,j]+=wpc.reshape(K,3,4)[:,:,j]
wmc=wmc.reshape(K,3,16)
b+=wo[:,:,-1].sum(1)+woc[:,:,-1].sum(1)+wmc[:,:,-1].sum(1)
x=np.concatenate([O[:,:,:-1].reshape(N,-1),Oc[:,:,:-1].reshape(N,-1),Mc.reshape(N,3,16)[:,:,:-1].reshape(N,-1)],1)
v=np.concatenate([(wo[:,:,:-1]-wo[:,:,-1,None]).reshape(K,-1),(woc[:,:,:-1]-woc[:,:,-1,None]).reshape(K,-1),(wmc[:,:,:-1]-wmc[:,:,-1,None]).reshape(K,-1)],1)
reduced=x@v.T+b
err=float(np.max(np.abs(original-reduced)));agreement=float(np.mean(original.argmax(1)==reduced.argmax(1)))
assert x.shape[1]==84 and err<1e-10 and agreement==1
combined={'raw_coordinates':112,'reduced_coordinates':84,'max_abs_score_error':err,'prediction_agreement':agreement}
results=[{'data_mode':'synthetic_algebra','configuration':'full_bank',**full},{'data_mode':'synthetic_algebra','configuration':'combined',**combined}]
p=Path(__file__).with_name('affine_reduction_checks.csv')
with p.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=results[0]);writer.writeheader();writer.writerows(results)
for row in results:print(row)
