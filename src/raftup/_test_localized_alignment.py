"""Frozen simulated localize/pilot/recovery path used by technology=test."""
import json,contextlib,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from . import _cutoff_selection as cs
from . import _recoverfull_new_new_knn, _fsgw_utils_test_sparse
CONFIG=dict(target_ratio=.067,min_points=150,max_points=800,max_ratio=.10,feature_down=.5,feature_full=.5,spatial_down=358.,spatial_full=274.,k=4)
def independently_cap_downsampling(a, b, shared_tar, sampled, target_ratio=0.067):
    """Keep a common pilot start, then increase each side's distance separately.

    The side with fewer spots retains at least 150; the larger side stays below
    10% whenever that bound is compatible with the 150-point floor.
    """
    da, db, ia, ib, _ = sampled
    sides = []
    for obj, ds, idx in ((a, da, ia), (b, db, ib)):
        n = len(obj)
        desired = min(n, max(150, int(np.ceil(target_ratio * n))))
        limit = min(n, 800, max(desired, int(np.floor(0.10 * n))))
        distance = int(shared_tar)
        if len(idx) > int(np.ceil(1.25 * desired)) or len(idx) > limit:
            proposed = int(round((distance * np.sqrt(len(idx) / max(desired, 1))) / 25) * 25)
            for trial in (proposed, proposed + 25, proposed - 25, proposed + 50, proposed - 50):
                trial = max(25, int(trial))
                np.random.seed(trial)
                attempt, attempt_idx = cs._downsample.downsample_slice(obj, trial)
                if attempt is None:
                    continue
                if 150 <= len(attempt_idx) <= limit and abs(len(attempt_idx) - desired) < abs(len(idx) - desired):
                    distance, ds, idx = trial, attempt, np.asarray(attempt_idx, dtype=int)
        sides.append((ds, np.asarray(idx, dtype=int), distance))
    (da, ia, ta), (db, ib, tb) = sides
    return da, db, ia, ib, (len(ia) / len(a), len(ib) / len(b)), (ta, tb)

def place(x,y,cost,ia,ib):
    center=y.mean(0);ys=y[ib]-center;tree=cKDTree(x)
    spacing=float(np.median(tree.query(x,k=2)[0][:,1]))
    centered=cost-cost.mean(1,keepdims=True)-cost.mean(0,keepdims=True)+cost.mean()
    def transform(p,points,reflection):
        c,s=np.cos(p[0]),np.sin(p[0])
        return (points*np.array([reflection,1]))@np.array([[c,s],[-s,c]])+p[1:]
    def objective(p,reflection):
        dist,idx=tree.query(transform(p,ys,reflection),k=3)
        v=centered[idx,ib[:,None]]+.15*(dist/spacing)**2
        choice=v.argmin(1);r=np.arange(len(ib));selected=idx[r,choice]
        return float(v[r,choice].mean()+.2*(1-len(np.unique(selected))/len(selected)))
    candidates=[]
    for reflection in (1,-1):
        for deg in range(0,360,10):
            for location in x[ia]:
                p=np.r_[np.deg2rad(deg),location];candidates.append((objective(p,reflection),reflection,p))
    candidates.sort(key=lambda z:z[0]);seeds=[]
    for item in candidates:
        if all(item[1]!=t[1] or np.linalg.norm(item[2][1:]-t[2][1:])>spacing*2 or abs(np.angle(np.exp(1j*(item[2][0]-t[2][0]))))>.2 for t in seeds):seeds.append(item)
        if len(seeds)==20:break
    refined=[]
    for _,reflection,p in seeds:
        result=minimize(lambda q:objective(q,reflection),p,method='Powell',bounds=[(p[0]-.25,p[0]+.25),(p[1]-2*spacing,p[1]+2*spacing),(p[2]-2*spacing,p[2]+2*spacing)],options={'maxiter':50,'xtol':.01})
        refined.append((float(result.fun),reflection,result.x))
    refined.sort(key=lambda z:z[0]);best=refined[0]
    return transform(best[2],y-center,best[1]),dict(objective=best[0],reflection=best[1],parameters=best[2].tolist(),spacing=spacing,candidates=[dict(score=v,reflection=r,parameters=p.tolist()) for v,r,p in refined])

def run(slice_a,slice_b,feature_cost,output_dir,*,sampling_indices=None):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    x=np.asarray(slice_a.obsm['spatial'],float);y=np.asarray(slice_b.obsm['spatial'],float);raw=np.asarray(feature_cost,float)
    if raw.shape!=(len(x),len(y)) or not np.isfinite(raw).all() or (raw<0).any():raise ValueError('Invalid feature-cost matrix')
    if sampling_indices is not None:
        ia,ib=(np.asarray(v,dtype=int) for v in sampling_indices)
        for v,n in ((ia,len(x)),(ib,len(y))):
            if len(v)==0 or len(np.unique(v))!=len(v) or np.any(v<0) or np.any(v>=n):raise ValueError('Invalid sampling indices')
        ratios=(len(ia)/len(x),len(ib)/len(y));distance=None
    elif max(len(x),len(y))<=150:
        ia,ib=np.arange(len(x)),np.arange(len(y));ratios=(1.,1.);distance=None
    else:
        shared,sampled=cs.choose_tar_distance(slice_a,slice_b,.067,min_points=150,max_points=800,max_ratio=.10,distance_step=25,seed=0)
        _,_,ia,ib,ratios,distance=independently_cap_downsampling(slice_a,slice_b,shared,sampled)
    np.savez(out/'indices.npz',idx_a=ia,idx_b=ib)
    scale=float(np.percentile(raw,99));norm=np.minimum(raw,scale)/max(scale,1e-12)
    z,placement=place(x,y,norm,ia,ib);np.save(out/'B_transformed.npy',z)
    decision=dict(config=CONFIG,sampled_A=len(ia),sampled_B=len(ib),ratios=ratios,tar_distance=distance,normalization_p99=scale,placement=placement,selection_uses_G_labels_LAA_GPR=False)
    (out/'decision.json').write_text(json.dumps(decision,indent=2))
    sub=norm[np.ix_(ia,ib)].copy();sub[cdist(x[ia],z[ib])>358]=2
    with (out/'solver.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        p=_fsgw_utils_test_sparse.fsgw_mvc_exp_sparse_heap(cdist(x[ia],x[ia]),cdist(y[ib],y[ib]),sub,gw_cutoff=358,w_cutoff=.5,show_progress=False)
        hard=cs._harden_rows(p);ra=np.flatnonzero(hard.any(1));cb=np.flatnonzero(hard.any(0))
        np.save(out/'downsampled_alignment.npy',hard)
        if len(ra)<4 or len(cb)<4:raise RuntimeError('Insufficient nonzero anchors; recovery refused')
        full=_recoverfull_new_new_knn.recover_full_mapping_knn(M=raw,X1=x,X2=y,P=hard[np.ix_(ra,cb)],idx1=ia[ra],idx2=ib[cb],pruned_P=True,k1=4,k2=4,thresh_CGW=274,thresh_CCC=.5,eps=.01).toarray()
    np.save(out/'before_distance_filter.npy',full)
    full[cdist(x,z)>274]=0
    np.save(out/'alignment_matrix.npy',full)
    decision.update(pilot_pairs=int((hard>0).sum()),active_A=len(ra),active_B=len(cb),seconds=time.perf_counter()-start)
    (out/'decision.json').write_text(json.dumps(decision,indent=2))
    return full,decision
