"""Prepare portable ordered tutorial inputs; never invent an existing cost's ID order.

Example: python scripts/prepare_revision_pair.py --a A.h5ad --b B.h5ad
 --output data/revision/my_pair --name my_pair --technology merfish --pixels-per-100um 100
Missing costs are computed using SCAN-IT. For an archived .npy cost, supply its two
cell-ID JSON files; these are checked against the input observation order.
"""
from pathlib import Path
import argparse,json
import anndata as ad
import numpy as np
from raftup.experiments import load_pair

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['a','b','output','name','technology']:p.add_argument('--'+key,required=True)
    p.add_argument('--pixels-per-100um',type=float,required=True)
    p.add_argument('--cost');p.add_argument('--ids-a');p.add_argument('--ids-b')
    p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=3500)
    args=p.parse_args()
    if args.technology not in {'dlpfc','merfish'}:p.error('technology must be dlpfc or merfish')
    output=Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use an empty output directory')
    a,b=ad.read_h5ad(args.a),ad.read_h5ad(args.b)
    for x in [a,b]:
        if not x.obs_names.is_unique or np.asarray(x.obsm['spatial']).shape!=(len(x),2):
            raise ValueError('Unique IDs and n-by-2 spatial coordinates are required')
    if args.cost:
        if not args.ids_a or not args.ids_b:p.error('Archived costs require --ids-a and --ids-b')
        for x,f in [(a,args.ids_a),(b,args.ids_b)]:
            if list(x.obs_names.astype(str))!=json.loads(Path(f).read_text()):raise ValueError('Cost/slice cell order mismatch')
        cost=np.load(args.cost,allow_pickle=False)
        cost_source='provided; ID order checked'
    else:
        from raftup._cutoff_selection import compute_scanit_cost
        aa,bb=a.copy(),b.copy()
        for x in [aa,bb]:x.obsm['spatial']=np.asarray(x.obsm['spatial'],float)*100/args.pixels_per_100um
        cost=compute_scanit_cost(aa,bb,technology=args.technology,n_epoch=args.epochs,seed=0,device=args.device)
        cost_source='new SCAN-IT cost; not an archived paper matrix'
    if cost.shape!=(len(a),len(b)) or not np.isfinite(cost).all() or (cost<0).any():raise ValueError('Invalid cost')
    output.mkdir(parents=True,exist_ok=True)
    a.write_h5ad(output/'slice_a.h5ad');b.write_h5ad(output/'slice_b.h5ad');np.save(output/'feature_cost.npy',cost)
    for x,name in [(a,'a'),(b,'b')]: (output/f'cell_ids_{name}.json').write_text(json.dumps(list(x.obs_names.astype(str))))
    manifest=dict(name=args.name,slice_a='slice_a.h5ad',slice_b='slice_b.h5ad',feature_cost='feature_cost.npy',
        cell_ids_a='cell_ids_a.json',cell_ids_b='cell_ids_b.json',data_technology=args.technology,
        pixels_per_100um=args.pixels_per_100um,cost_source=cost_source)
    (output/'pair.json').write_text(json.dumps(manifest,indent=2))
    load_pair(output/'pair.json')
    print(output/'pair.json')
if __name__=='__main__':main()
