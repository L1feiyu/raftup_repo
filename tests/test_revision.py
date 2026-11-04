import unittest
import tempfile,json
from pathlib import Path
import numpy as np
from scipy import sparse
from raftup.experiments import coverage, rigid_from_alignment, purity_concordance, spatial_distortion, tissue_nrmse, load_pair

class RevisionTests(unittest.TestCase):
    def test_known_rigid_transform(self):
        y=np.array([[0.,0.],[1.,0.],[0.,2.],[2.,3.]])
        R=np.array([[0.,-1.],[1.,0.]]);t=np.array([4.,7.]);x=y@R+t
        r,z=rigid_from_alignment(sparse.eye(4),x,y)
        np.testing.assert_allclose(y@r+z,x,atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r),1)
        self.assertLess(tissue_nrmse(y@r+z,x,10),1e-12)
    def test_mnn_excludes_distant_cells(self):
        x=np.array([[0.,0.],[10.,0.],[20.,0.]])
        y=np.array([[0.,0.],[10.,0.],[200.,0.]])
        result,i,j=purity_concordance(x,y,[.1,.5,.9],[.1,.5,.9],1)
        self.assertEqual(result['n_pairs'],2)
        self.assertAlmostEqual(result['purity_MAE'],0)
        self.assertAlmostEqual(result['purity_spearman'],1)
    def test_all_support_pairs_include_shared_endpoints(self):
        P=np.array([[1.,1.],[0.,1.]])
        x=np.array([[0.,0.],[5.,0.]]);y=np.array([[0.,0.],[25.,0.]])
        report=spatial_distortion(P,x,y,10,block_size=1)
        self.assertEqual(report['total_combinations'],3)
        self.assertEqual(report['violating_combinations'],2)
        self.assertEqual(report['excess_counts'],[0,2,0,0,0])
        self.assertEqual(coverage(P),{'coverage_A':1.,'coverage_B':1.})
    def test_input_order_validation(self):
        import anndata as ad
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);a=ad.AnnData(np.eye(3));a.obsm['spatial']=np.zeros((3,2))
            a.write_h5ad(d/'a.h5ad');a.write_h5ad(d/'b.h5ad');np.save(d/'c.npy',np.eye(3))
            ids=list(a.obs_names);(d/'ids.json').write_text(json.dumps(ids))
            cfg=dict(slice_a='a.h5ad',slice_b='b.h5ad',feature_cost='c.npy',cell_ids_a='ids.json',cell_ids_b='ids.json')
            (d/'pair.json').write_text(json.dumps(cfg));load_pair(d/'pair.json')
            (d/'ids.json').write_text(json.dumps(ids[::-1]))
            with self.assertRaises(ValueError):load_pair(d/'pair.json')

if __name__=='__main__':unittest.main()
