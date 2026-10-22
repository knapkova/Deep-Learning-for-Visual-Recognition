"""完整小矩阵的独立审计，以及概率/标签/训练角色/缺件反例。"""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from audit_results import METHODS, REFERENCE, _ids_digest, _read_csv, _select, audit_results


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class AuditResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results = self.root/'results'
        self.manifest = self.root/'manifest.csv'
        self.config = dict(sources=[0,1], depths=['head'], seeds=[42,43], epochs=2,
                           sampling_seed=42, adaptation_sources=[0,1], adaptation_targets=[0,1],
                           adaptation_seeds=[42], max_train_per_class=None,
                           max_val_per_class=None, max_test_per_class=None)
        rows = []
        for center in ['0','1']:
            for partition in ['train','val','test']:
                for label in [0,1]:
                    sid = f'{center}-{partition}-{label}'
                    rows.append(dict(sample_id=sid,path=sid+'.png',center=center,
                                     patient=f'{center}-{partition}',slide=f'{center}-{partition}',
                                     label=label,partition=partition))
        self.frame = pd.DataFrame(rows)
        self.frame.to_csv(self.manifest,index=False)
        dump(self.results/'config.json',self.config)
        for source in self.config['sources']:
            for seed in self.config['seeds']:
                self.make_run(source,seed)
        self.make_adaptation(self.results/'analysis',[0,1],REFERENCE)
        for source in [0,1]:
            self.make_adaptation(self.results/'depth_adaptation'/f'source_{source}'/'head'/'seed_42',
                                 [source],f'source_{source}/head/seed_42')

    def tearDown(self):
        self.temporary.cleanup()

    def make_run(self,source,seed):
        folder=self.results/'runs'/f'source_{source}'/'head'/f'seed_{seed}'
        folder.mkdir(parents=True)
        train=_select(self.frame,'train',source)
        val=_select(self.frame,'val',source)
        test=_select(self.frame,'test')
        metrics=dict(config=self.config,source=str(source),depth='head',seed=seed,
                     status='complete',reload_verified=True,train_n=2,val_n=2,test_n=4,
                     dataset_digest=hashlib.sha256(str(source).encode()).hexdigest(),
                     initial_state_digest=hashlib.sha256(str(seed).encode()).hexdigest(),
                     sampling={f'{key}_sample_ids_digest':_ids_digest(split)
                               for key,split in [('train',train),('val',val),('test',test)]},
                     selection=dict(split='source_val',center=str(source),target_test_used_for_selection=False,
                                    epochs_completed=2,best_epoch=1,best_val_loss=.5),
                     reloaded_validation={'loss':.5},
                     test_by_center={str(c):dict(n=2,accuracy=1.,roc_auc=1.,role='ID' if c==source else 'OOD') for c in [0,1]})
        dump(folder/'metrics.json',metrics)
        pd.DataFrame([dict(epoch=1,val_loss=.5),dict(epoch=2,val_loss=.6)]).to_csv(folder/'history.csv',index=False)
        predictions=test[['sample_id','center','label','partition']].copy()
        predictions['source']=source
        predictions['probability']=np.where(predictions['label'].eq(1),.8,.2)
        predictions.to_csv(folder/'predictions.csv',index=False)

    def make_adaptation(self,folder,sources,feature_source):
        folder.mkdir(parents=True,exist_ok=True)
        predictions,results,pairs=[],[],[]
        def ids(center,partition):
            return self.frame.loc[self.frame.center.eq(str(center)) & self.frame.partition.eq(partition),'sample_id'].tolist()
        loss=float(log_loss([0,1],[[.8,.2],[.2,.8]],labels=[0,1]))
        history=[dict(C=.1,accuracy=1.,roc_auc=1.,log_loss=.2),dict(C=1.,accuracy=1.,roc_auc=1.,log_loss=.3)]
        for source in sources:
            for target in [0,1]:
                if source==target:
                    continue
                pairs.append(dict(source=str(source),target=str(target),projection_fit_sample_ids=ids(source,'train'),
                                  source_labeled_fit_sample_ids=ids(source,'train'),target_unlabeled_fit_sample_ids=ids(target,'train'),
                                  selection_sample_ids=ids(source,'val'),evaluation_sample_ids=ids(target,'test'),
                                  baseline_C_source_val=history,coral_C_source_val=history))
                for method in METHODS:
                    results.append(dict(source=source,target=target,method=method,feature_source=feature_source,
                                        target_test_n=2,accuracy=1.,balanced_accuracy=1.,roc_auc=1.,log_loss=loss,
                                        selected_C=.1,source_val_accuracy=1.,source_val_roc_auc=1.))
                    for row in self.frame.loc[self.frame.center.eq(str(target)) & self.frame.partition.eq('test')].to_dict('records'):
                        predictions.append({**{k:v for k,v in row.items() if k!='path'},'source':source,'target':target,
                                            'method':method,'feature_source':feature_source,
                                            'probability':.8 if row['label'] else .2,'prediction':row['label']})
        dump(folder/'adaptation_metadata.json',dict(feature_source=feature_source,seed=42,
             target_labels_used_for_fitting=False,target_test_features_used_for_fitting=False,pairs=pairs))
        pd.DataFrame(results).to_csv(folder/'adaptation_results.csv',index=False)
        pd.DataFrame(predictions).to_csv(folder/'adaptation_predictions.csv',index=False)

    def audit(self):
        return audit_results(self.manifest,self.results,self.config)

    def test_complete_matrix_and_adaptation_pass_without_checkpoint_bytes(self):
        report=self.audit()
        self.assertEqual(report['failures'],[])
        self.assertTrue(report['complete'])
        self.assertEqual(report['checked_runs'],4)
        self.assertEqual(report['patch_predictions'],16)
        self.assertEqual(report['adaptation_pairs'],4)
        self.assertEqual(report['adaptation_predictions'],16)
        self.assertFalse(report['checkpoint_bytes_verified_by_this_audit'])
        self.assertTrue((self.results/'audit.json').exists())

    def test_main_only_passes_without_adaptation_and_preserves_full_audit_receipt(self):
        self.assertTrue(self.audit()['complete'])
        original = (self.results/'audit.json').read_bytes()
        shutil.rmtree(self.results/'analysis')
        shutil.rmtree(self.results/'depth_adaptation')
        report = audit_results(self.manifest,self.results,self.config,main_only=True)
        self.assertTrue(report['complete'])
        self.assertEqual(report['checked_runs'],4)
        self.assertEqual(report['adaptation_pairs'],0)
        self.assertEqual(report['scope'],'main project only')
        self.assertTrue((self.results/'audit_main.json').exists())
        self.assertEqual((self.results/'audit.json').read_bytes(),original)

    def test_bad_probability_and_manifest_label_are_rejected(self):
        path=self.results/'runs/source_0/head/seed_42/predictions.csv'
        original=pd.read_csv(path)
        for column,value in [('probability',-0.1),('label',1)]:
            with self.subTest(column=column):
                changed=original.copy()
                changed.loc[0,column]=value
                changed.to_csv(path,index=False)
                report=self.audit()
                self.assertFalse(report['complete'])
                self.assertTrue(any('Prediction label' in f['error'] or 'Probability outside' in f['error'] for f in report['failures']))
        original.to_csv(path,index=False)

    def test_near_one_probability_round_trips_without_log_loss_drift(self):
        # Regression from the real 4->2 CORAL export: 0.9999999999999987 was
        # previously read as 0.9999999999999988 by the default pandas C parser.
        folder=self.results/'analysis'
        prediction_path=folder/'adaptation_predictions.csv'
        predictions=pd.read_csv(prediction_path,float_precision='round_trip')
        group=predictions.source.eq(0) & predictions.target.eq(1) & predictions.method.eq('coral_logistic')
        positive=np.array([0.9999999999999987,.8],dtype=np.float64)
        labels=np.array([0,1])
        predictions.loc[group,'probability']=positive
        predictions.loc[group,'prediction']=(positive>=.5).astype(int)
        predictions.to_csv(prediction_path,index=False)
        restored=_read_csv(prediction_path).loc[group,'probability'].to_numpy()
        np.testing.assert_array_equal(restored,positive)
        expected_loss=float(log_loss(labels,np.column_stack([1-positive,positive]),labels=[0,1]))
        result_path=folder/'adaptation_results.csv'
        results=pd.read_csv(result_path,float_precision='round_trip')
        row=results.source.eq(0) & results.target.eq(1) & results.method.eq('coral_logistic')
        results.loc[row,['accuracy','balanced_accuracy','roc_auc','log_loss']]=[.5,.5,0.,expected_loss]
        results.to_csv(result_path,index=False)
        report=self.audit()
        self.assertEqual(report['failures'],[])
        self.assertTrue(report['complete'])
        self.assertEqual(report['metric_tolerance'],1e-6)

    def test_partial_matrix_is_not_a_pass(self):
        (self.results/'runs/source_1/head/seed_43/metrics.json').unlink()
        report=self.audit()
        self.assertFalse(report['complete'])
        self.assertEqual(report['checked_runs'],3)
        self.assertTrue(any(f['assertion']=='exact configured run matrix' for f in report['failures']))

    def test_adaptation_test_fit_id_and_wrong_metric_are_rejected(self):
        metadata_path=self.results/'analysis/adaptation_metadata.json'
        metadata=json.loads(metadata_path.read_text())
        metadata['pairs'][0]['target_unlabeled_fit_sample_ids'].append(metadata['pairs'][0]['evaluation_sample_ids'][0])
        dump(metadata_path,metadata)
        report=self.audit()
        self.assertFalse(report['complete'])
        self.assertTrue(any('roles/IDs' in f['error'] for f in report['failures']))
        # Independently modified metric is caught even with valid roles restored.
        self.make_adaptation(self.results/'analysis',[0,1],REFERENCE)
        path=self.results/'analysis/adaptation_results.csv'
        results=pd.read_csv(path)
        results.loc[0,'roc_auc']=.5
        results.to_csv(path,index=False)
        report=self.audit()
        self.assertFalse(report['complete'])
        self.assertTrue(any('independently recomputed' in f['error'] for f in report['failures']))

    def test_patient_leakage_and_wrong_checkpoint_epoch_are_rejected(self):
        path=self.results/'runs/source_0/head/seed_42/metrics.json'
        metrics=json.loads(path.read_text())
        metrics['selection']['best_epoch']=2
        dump(path,metrics)
        self.assertTrue(any('minimum source-val' in f['error'] for f in self.audit()['failures']))
        changed=self.frame.copy()
        changed.loc[changed.center.eq('0'),'patient']='leaked'
        changed.to_csv(self.manifest,index=False)
        report=self.audit()
        self.assertFalse(report['complete'])
        self.assertTrue(any('patient leakage' in f['error'] for f in report['failures']))


if __name__=='__main__':
    unittest.main()
