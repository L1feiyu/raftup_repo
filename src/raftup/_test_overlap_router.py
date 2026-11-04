"""Feature-informed geometry router. Development parameters, not independently validated."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
def spacing(x):return float(np.median(cKDTree(x).query(x,k=2)[0][:,1]))

def matrix(theta,reflection):
 c,s=np.cos(theta),np.sin(theta);return np.diag([reflection,1])@np.array([[c,s],[-s,c]])

def fit(x,y,centered,seed):
 rng=np.random.default_rng(seed);iy=np.sort(rng.choice(len(y),min(160,len(y)),replace=False));ix=np.sort(rng.choice(len(x),min(80,len(x)),replace=False))
 sy=spacing(y);sx=spacing(x);scale=max((sx+sy)/2,1e-12);center=y.mean(0);ys=y[iy]-center;tree=cKDTree(x)
 def obj(p,ref):
  dist,idx=tree.query(ys@matrix(p[0],ref)+p[1:],k=3)
  values=centered[idx,iy[:,None]]+.15*(dist/scale)**2
  # Every query can abstain with zero cost. No reward for matching poor candidates.
  return float(np.minimum(values.min(1),0.).mean())
 candidates=[]
 for ref in (1,-1):
  for angle in np.deg2rad(np.arange(0,360,15)):
   for location in x[ix]:
    p=np.r_[angle,location];candidates.append((obj(p,ref),ref,p))
 candidates.sort(key=lambda a:a[0]);seeds=[]
 for item in candidates:
  if all(item[1]!=t[1] or np.linalg.norm(item[2][1:]-t[2][1:])>2*scale or abs(np.angle(np.exp(1j*(item[2][0]-t[2][0]))))>.25 for t in seeds):seeds.append(item)
  if len(seeds)==8:break
 solutions=[]
 for val,ref,p in seeds:
  res=minimize(lambda q:obj(q,ref),p,method='Powell',bounds=[(p[0]-.3,p[0]+.3),(p[1]-2*scale,p[1]+2*scale),(p[2]-2*scale,p[2]+2*scale)],options={'maxiter':35,'xtol':.02})
  solutions.extend([(val,ref,p),(float(res.fun),ref,res.x)])
 score,ref,p=min(solutions,key=lambda a:a[0]);r=matrix(p[0],ref);t=p[1:]-center@r
 return r,t,score,scale

def route_coordinates(x,y,feature_cost,seeds=(11,29,47)):
 x,y=np.asarray(x,float),np.asarray(y,float);raw=np.asarray(feature_cost,float)
 if raw.shape!=(len(x),len(y)) or min(len(x),len(y))<3 or not np.isfinite(raw).all():raise ValueError('Invalid coordinates/cost dimensions')
 q=float(np.percentile(raw,99));c=np.minimum(raw,q)/max(q,1e-12);c=c-c.mean(1,keepdims=True)-c.mean(0,keepdims=True)+c.mean()
 s=(spacing(x)+spacing(y))/2;reps=[]
 for seed in seeds:
  candidates=[]
  for reverse in (False,True):
   r,t,objective,_=fit(y,x,c.T,seed) if reverse else fit(x,y,c,seed)
   if reverse:r,t=r.T,-t@r.T
   z=y@r+t
   ga=float((cKDTree(z).query(x)[0]<=2*s).mean());gb=float((cKDTree(x).query(z)[0]<=2*s).mean())
   candidates.append(dict(direction='A_to_B_inverted' if reverse else 'B_to_A',geometry_A=ga,geometry_B=gb,support=max(ga,gb),objective=objective,R=r.tolist(),t=t.tolist()))
  best=max(candidates,key=lambda d:d['support']);reps.append(dict(seed=int(seed),best_support=best['support'],passes=best['support']>=.98,candidates=candidates))
 if len(seeds)!=3:raise ValueError('Development rule requires exactly three repetitions')
 successes=sum(r['passes'] for r in reps)
 return dict(route='simulated_localized' if successes>=2 else 'synthetic_window',successes=successes,repetitions=reps,geometry_radius_spacing=2.,coverage_threshold=.98,required_successes=2,development_only=True,uses_ground_truth=False)

def route_slices(slice_a,slice_b,feature_cost,seeds=(11,29,47)):
 return route_coordinates(slice_a.obsm['spatial'],slice_b.obsm['spatial'],feature_cost,seeds)
