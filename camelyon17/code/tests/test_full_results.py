"""完整/缺件边界、配对改善及模型seed/患者split两层变化。"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from full_results import _score,generate


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class FullResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.args=dict(split_seeds=(42,43),sources=(0,1),depths=('head','full'),model_seeds=(42,43),expected_images=16)
        for split in [42,43]:
            folder=self.root/f'split_{split}'
            config=dict(sources=[0,1],depths=['head','full'],seeds=[42,43],epochs=20,
                        early_stopping_patience=5,partition_seed=split,sampling_seed=42,per_slide_class=None,
                        max_train_per_class=None,max_val_per_class=None,max_test_per_class=None)
            write_json(folder/'config.json',config)
            write_json(folder/'provenance.json',dict(selected_images=16,per_slide_class_cap=None,fixture_per_center=None))
            (folder/'analysis').mkdir()
            pd.DataFrame([dict(source=s,target=t,shift_raw=.1+(split-42)*.01,
                               shift_balanced=.2,shift_classconditional=.3)
                          for s,t in [(0,1),(1,0)]]).to_csv(folder/'analysis/representation_shift.csv',index=False)
            for source in [0,1]:
                for depth in ['head','full']:
                    for seed in [42,43]:self.make_run(folder,config,split,source,depth,seed)

    def tearDown(self):self.temp.cleanup()

    def make_run(self,folder,config,split,source,depth,seed):
        path=folder/'runs'/f'source_{source}'/depth/f'seed_{seed}';path.mkdir(parents=True)
        labels=np.array([0,0,1,1])
        if depth=='head':probs=np.array([.1,.6,.4,.9]) if seed==42 else np.array([.1,.8,.4,.6])
        else:probs=np.array([.1,.2,.8,.9]) if seed==42 else np.array([.1,.6,.8,.9])
        if split==43 and depth=='full':probs=np.clip(probs+np.array([.05,.05,-.05,-.05]),0,1)
        target_metrics={};pred_rows=[]
        for target in [0,1]:
            computed=_score(labels,probs)
            loss=float(-np.log(np.where(labels,probs,1-probs)).mean())
            target_metrics[str(target)]={**computed,'loss':loss,'role':'ID' if source==target else 'OOD'}
            pred_rows.extend(dict(sample_id=f'c{target}-test-{i}',center=target,label=int(y),partition='test',source=source,probability=float(p)) for i,(y,p) in enumerate(zip(labels,probs)))
        predictions=pd.DataFrame(pred_rows).sort_values('sample_id');predictions.to_csv(path/'predictions.csv',index=False)
        h=pd.DataFrame(dict(epoch=range(1,7),train_loss=np.linspace(.8,.2,6),val_loss=np.arange(6)*.1+.4))
        h.to_csv(path/'history.csv',index=False)
        metrics=dict(config=config,status='complete',reload_verified=True,source=str(source),depth=depth,seed=seed,
                     train_n=2,val_n=2,test_n=8,test_by_center=target_metrics,
                     dataset_digest=f'{split}-{source}',initial_state_digest=hashlib.sha256(str(seed).encode()).hexdigest(),
                     sampling=dict(train_sample_ids_digest=f'{split}-{source}-train',val_sample_ids_digest=f'{split}-{source}-val',
                                   test_sample_ids_digest=hashlib.sha256('\n'.join(predictions.sample_id).encode()).hexdigest()),
                     selection=dict(split='source_val',center=str(source),target_test_used_for_selection=False,best_epoch=1,best_val_loss=.4,epochs_completed=6),
                     reloaded_validation={'loss':.4})
        write_json(path/'metrics.json',metrics)

    def run_report(self):
        with patch('full_results._plot_matrices',return_value=[]):return generate(self.root,**self.args)

    def test_complete_count_paired_deltas_and_separate_variability(self):
        status=self.run_report()
        self.assertEqual(status['issues'],[])
        self.assertTrue(status['complete'])
        self.assertEqual(status['complete_runs'],16)
        self.assertEqual(status['hospital_metric_rows'],32)
        delta=pd.read_csv(self.root/'full_depth_minus_head_per_run.csv')
        self.assertEqual(len(delta),16)
        np.testing.assert_allclose(sorted(delta.roc_auc_delta.unique()),[.25,.5])
        summary=pd.read_csv(self.root/'full_depth_minus_head_across_splits.csv')
        np.testing.assert_allclose(summary.roc_auc_delta_mean,.375)
        np.testing.assert_allclose(summary.roc_auc_delta_between_patient_split_variance,0.)
        np.testing.assert_allclose(summary.roc_auc_delta_mean_within_split_model_seed_variance,.03125)
        np.testing.assert_allclose(summary.roc_auc_delta_between_patient_split_std,0.)
        np.testing.assert_allclose(summary.roc_auc_delta_mean_within_split_model_seed_std,np.std([.25,.5],ddof=1))
        variability=pd.read_csv(self.root/'full_across_patient_split_summary.csv')
        self.assertTrue((variability.loc[variability.depth.eq('full'),'loss_between_patient_split_std']>0).all())
        winners=pd.read_csv(self.root/'full_winner_frequency_by_target.csv')
        auc=winners[winners.metric.eq('roc_auc')]
        self.assertTrue(auc.loc[auc.depth.eq('full'),'observed_frequency'].eq(1).all())
        self.assertTrue(auc.loc[auc.depth.eq('head'),'observed_frequency'].eq(0).all())
        self.assertTrue(auc.complete_source_split_comparisons.eq(2).all())
        baseline=pd.read_csv(self.root/'full_test_class_baselines.csv')
        self.assertTrue(baseline.majority_accuracy_baseline.eq(.5).all())
        self.assertEqual(baseline.positive_n.sum(),8)
        page=(self.root/'full_project_report.html').read_text()
        self.assertIn('16 / 16',page)
        self.assertIn('不把 patch 当独立患者',page)

    def test_missing_run_keeps_prespecified_denominator_and_excludes_incomplete_winner(self):
        (self.root/'split_43/runs/source_0/full/seed_43/predictions.csv').unlink()
        status=self.run_report()
        self.assertFalse(status['complete'])
        self.assertEqual(status['expected_runs'],16)
        self.assertEqual(status['complete_runs'],15)
        self.assertEqual(status['missing_runs'],1)
        self.assertIn('未完整（partial）',(self.root/'full_project_report.html').read_text())
        winners=pd.read_csv(self.root/'full_observed_winners.csv')
        self.assertTrue(winners.loc[winners.split_seed.eq(43)&winners.source.eq(0)].empty)

    def test_bad_probability_and_wrong_full_coverage_are_not_complete(self):
        p=self.root/'split_42/runs/source_0/full/seed_42/predictions.csv'
        frame=pd.read_csv(p);frame.loc[0,'probability']=1.1;frame.to_csv(p,index=False)
        write_json(self.root/'split_43/provenance.json',dict(selected_images=15,per_slide_class_cap=None))
        status=self.run_report()
        self.assertFalse(status['complete'])
        self.assertEqual(status['invalid_runs'],1)
        self.assertFalse(status['dataset_coverage'][1]['verified'])
        self.assertTrue(any('Invalid probability' in item['error'] for item in status['issues']))

    def test_rebuilding_partial_snapshot_does_not_link_stale_complete_metrics(self):
        self.assertTrue(self.run_report()['complete'])
        self.assertTrue((self.root/'full_metrics_all.csv').exists())
        for path in self.root.glob('split_*/runs/source_*/*/seed_*/metrics.json'):path.unlink()
        status=self.run_report()
        self.assertFalse(status['complete'])
        self.assertFalse((self.root/'full_metrics_all.csv').exists())
        page=(self.root/'full_project_report.html').read_text()
        self.assertNotIn('href="full_metrics_all.csv"',page)

    def test_no_results_cannot_shrink_full_default_180_plan(self):
        with tempfile.TemporaryDirectory() as empty:
            with patch('full_results._plot_matrices',return_value=[]):status=generate(empty)
            self.assertEqual(status['expected_runs'],180)
            self.assertEqual(status['complete_runs'],0)
            self.assertEqual(status['missing_runs'],180)
            self.assertFalse(status['complete'])


if __name__=='__main__':unittest.main()
