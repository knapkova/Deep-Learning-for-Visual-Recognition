import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from full_analysis import compute_full_shift


class FullShiftTests(unittest.TestCase):
    def test_all_train_members_exact_weighting_and_no_test_influence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            frame=pd.DataFrame({'sample_id':[str(i) for i in range(8)],
                'path':[f'{i}.png' for i in range(8)],'center':[0]*4+[1]*4,
                'patient':[0,0,0,2,3,3,3,5],'slide':[0,0,0,2,3,3,3,5],
                'label':[0,0,1,1,0,1,1,1],'partition':['train']*3+['test']+['train']*3+['test']})
            base=np.array([0,0,2,10000,1,3,3,-10000],dtype=np.float32)
            x=np.column_stack([base,2*base])
            np.save(p/'f.npy',x)
            frame.to_csv(p/'manifest.csv',index=False)
            frame.drop(columns='partition').to_csv(p/'rows.csv',index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                path=compute_full_shift(p/'f.npy',p/'rows.csv',p/'manifest.csv',p/'out')
            result=pd.read_csv(path)
            self.assertEqual(len(result),2)
            np.testing.assert_allclose(result.shift_raw,2.5)
            np.testing.assert_allclose(result.shift_balanced,1.5)
            np.testing.assert_allclose(result.shift_classconditional,1.5)
            self.assertTrue(result.source_train_n.eq(3).all())
            self.assertIsNone(json.loads((p/'out/representation_shift.metadata.json').read_text())['sample_cap'])
            x[[3,7]]=0
            np.save(p/'f.npy',x)
            with contextlib.redirect_stdout(io.StringIO()):
                other=compute_full_shift(p/'f.npy',p/'rows.csv',p/'manifest.csv',p/'out2')
            np.testing.assert_allclose(pd.read_csv(other).shift_raw,result.shift_raw)
            with self.assertRaises(ValueError):
                compute_full_shift(p/'f.npy',p/'rows.csv',p/'manifest.csv',p/'out')

    def test_plan_executes_all_sources_splits_seeds_depths_and_no_adaptation(self):
        import full_run
        config={'sources':list(range(5)),'partition_seeds':[42,43,44],'seeds':[42,43,44],
                'depths':['head','layer4','layer3_layer4','full']}
        events=[]
        def train(root,config,source,depth,seed,folder):
            events.append(('train',source,config['partition_seed'],depth,seed))
        def distance(*args):events.append(('distance',))
        with patch('camelyon.train.run_experiment',side_effect=train), \
             patch('full_analysis.extract_full_reference',return_value=('f.npy','rows.csv')) as extract, \
             patch('full_analysis.compute_full_shift',side_effect=distance), \
             patch('full_run.status'),patch('full_run.summarize'):
            full_run.train_and_analyze(Path('/tmp'),config,Path('/tmp/out'))
        runs=[x for x in events if x[0]=='train']
        self.assertEqual(len(runs),180)
        self.assertEqual(len(set(runs)),180)
        self.assertEqual([x[1] for x in runs],sum(([i]*36 for i in range(5)),[]))
        self.assertEqual(events[12],('distance',))
        extract.assert_called_once()


if __name__=='__main__':unittest.main()
