#!/usr/bin/env python3
"""独立核对下载后的完整结果；不读取模型权重、不运行推理或训练。

python audit_results.py --manifest data/manifest.csv --results results/colab_sampled
缺失实验、无效预测或元数据不一致均返回非零；部分结果不会标为通过。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
METHODS = {'source_only_logistic', 'coral_logistic'}
REFERENCE = 'torchvision.resnet50.IMAGENET1K_V2.avgpool2048'
META = ['sample_id', 'center', 'patient', 'slide', 'label', 'partition']


def _read_csv(path):
    # Saved float64 probabilities can be within a few ULPs of 1.  The default
    # C parser may move a value by one ULP, greatly changing -log(1-p).
    # Round-trip parsing restores the exact saved binary float; keep metric
    # tolerances unchanged rather than masking a CSV parsing error.
    return pd.read_csv(path, dtype={field: str for field in META if field != 'label'},
                       float_precision='round_trip')


def _select(frame, partition, source=None, cap=None, seed=42):
    # Independent reimplementation of the declared sampling protocol; no training imports.
    chosen = frame.loc[frame['partition'].eq(partition)]
    if source is not None:
        chosen = chosen.loc[chosen['center'].eq(str(source))]
    chosen = chosen.sort_values('sample_id', kind='stable')
    if cap is not None:
        assert int(cap) > 0, 'Per-class cap must be positive'
        groups = [group.sample(min(len(group), int(cap)), random_state=int(seed))
                  for _, group in chosen.groupby(['center', 'label'], sort=True)]
        chosen = pd.concat(groups) if groups else chosen
    return chosen.sort_values('sample_id', kind='stable').reset_index(drop=True)


def _ids_digest(frame):
    return hashlib.sha256('\n'.join(frame['sample_id']).encode()).hexdigest()


def _same_number(actual, expected, name, tolerance=1e-6):
    if expected is None:
        assert actual is None or pd.isna(actual), f'{name}: expected undefined metric, got {actual}'
    else:
        assert actual is not None and np.isfinite(float(actual)), f'{name}: nonfinite/missing value'
        assert np.isclose(float(actual), expected, atol=tolerance, rtol=0), \
            f'{name}: recorded {actual}, independently recomputed {expected}'


def _score(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    assert len(labels) > 0, 'Empty prediction split'
    assert np.isfinite(probabilities).all(), 'Probabilities must be finite'
    assert ((probabilities >= 0) & (probabilities <= 1)).all(), 'Probability outside [0, 1]'
    assert set(np.unique(labels)).issubset({0, 1}), 'Nonbinary prediction label'
    return {'n': len(labels), 'accuracy': float(accuracy_score(labels, probabilities >= .5)),
            'roc_auc': float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None}


def _bind_predictions(predictions, expected, fields):
    assert not predictions['sample_id'].duplicated().any(), 'Duplicate prediction sample_id'
    assert predictions['sample_id'].tolist() == expected['sample_id'].tolist(), \
        'Prediction IDs/order differ from expected test manifest'
    for field in fields:
        actual = predictions[field].astype(int if field == 'label' else str).to_numpy()
        wanted = expected[field].astype(int if field == 'label' else str).to_numpy()
        assert np.array_equal(actual, wanted), f'Prediction {field} differs from manifest'
    _score(predictions['label'], predictions['probability'])


class Audit:
    def __init__(self, manifest, results, config, main_only=False):
        self.manifest_path, self.results = Path(manifest), Path(results)
        self.config = config
        self.main_only = main_only
        self.failures, self.assertions = [], []
        self.runs, self.patch_predictions, self.adaptation_pairs = 0, 0, 0
        self.adaptation_methods, self.adaptation_predictions = 0, 0
        self.records = []

    def check(self, path, label, function):
        try:
            function()
            self.assertions.append({'path': str(path), 'assertion': label, 'passed': True})
        except Exception as error:
            self.failures.append({'path': str(path), 'assertion': label,
                                  'error': f'{type(error).__name__}: {error}'})

    def manifest(self):
        frame = _read_csv(self.manifest_path)
        assert set(META + ['path']).issubset(frame.columns), 'Missing manifest columns'
        assert len(frame) and not frame[META + ['path']].isna().any().any(), 'Empty/null manifest'
        assert not frame['sample_id'].duplicated().any(), 'Duplicate manifest sample_id'
        assert not frame['path'].duplicated().any(), 'Duplicate manifest image path'
        assert set(frame['label'].unique()).issubset({0, 1}), 'Nonbinary manifest labels'
        assert set(frame['partition']).issubset({'train', 'val', 'test'}), 'Unknown partition'
        for group in ['patient', 'slide']:
            assert not (frame.groupby(['center', group])['partition'].nunique() > 1).any(), \
                f'{group} leakage across train/val/test'
        self.frame = frame

    def run(self, source, depth, seed):
        path = self.results / 'runs' / f'source_{source}' / depth / f'seed_{seed}'
        metrics = json.loads((path / 'metrics.json').read_text())
        assert metrics['config'] == self.config, 'Run config differs from expected full config'
        assert str(metrics['source']) == str(source) and metrics['depth'] == depth and metrics['seed'] == seed, 'Wrong run identity'
        assert metrics['status'] == 'complete' and metrics['reload_verified'] is True, 'Incomplete/unverified reload'
        assert isinstance(metrics['initial_state_digest'], str) and len(metrics['initial_state_digest']) == 64, 'Invalid initial digest'
        sampling_seed = self.config.get('sampling_seed', 42)
        train = _select(self.frame, 'train', source, self.config.get('max_train_per_class'), sampling_seed)
        val = _select(self.frame, 'val', source, self.config.get('max_val_per_class'), sampling_seed)
        test = _select(self.frame, 'test', None, self.config.get('max_test_per_class'), sampling_seed)
        assert metrics['train_n'] == len(train) and metrics['val_n'] == len(val) and metrics['test_n'] == len(test), 'Incorrect split counts'
        for partition, expected in [('train', train), ('val', val), ('test', test)]:
            assert metrics['sampling'][f'{partition}_sample_ids_digest'] == _ids_digest(expected), f'{partition} input digest mismatch'
        history = pd.read_csv(path / 'history.csv', float_precision='round_trip')
        assert len(history) and np.isfinite(history['val_loss']).all(), 'Invalid validation history'
        assert history['epoch'].tolist() == list(range(1, len(history) + 1)), 'History epoch gaps/duplicates'
        assert len(history) <= self.config['epochs'], 'More epochs than configured'
        if self.config.get('early_stopping_patience') is None:
            assert len(history) == self.config['epochs'], 'Full configured epoch budget not completed'
        selection = metrics['selection']
        assert selection['split'] == 'source_val' and str(selection['center']) == str(source), 'Selection is not source validation'
        assert selection['target_test_used_for_selection'] is False, 'Target test used for selection'
        assert selection['epochs_completed'] == len(history), 'Epoch count metadata mismatch'
        selected = history.iloc[int(np.argmin(history['val_loss'].to_numpy()))]
        assert selection['best_epoch'] == int(selected['epoch']), 'Selected epoch is not first minimum source-val loss'
        _same_number(selection['best_val_loss'], float(selected['val_loss']), 'selected source-val loss')
        _same_number(metrics['reloaded_validation']['loss'], float(selected['val_loss']), 'reloaded source-val loss')
        predictions = _read_csv(path / 'predictions.csv')
        _bind_predictions(predictions, test, ['center', 'label', 'partition'])
        assert predictions['source'].astype(str).eq(str(source)).all(), 'Prediction source differs from run'
        assert set(metrics['test_by_center']) == set(test['center']), 'Per-center metric coverage differs'
        for center, group in predictions.groupby('center'):
            computed = _score(group['label'], group['probability'])
            recorded = metrics['test_by_center'][str(center)]
            for name, value in computed.items():
                _same_number(recorded[name], value, f'center {center} {name}')
            assert recorded['role'] == ('ID' if str(center) == str(source) else 'OOD'), 'Incorrect ID/OOD role'
        self.records.append(metrics)
        self.runs += 1
        self.patch_predictions += len(predictions)

    def common_inputs_and_initialization(self):
        assert self.records, 'No valid runs to compare'
        for source in self.config['sources']:
            records = [r for r in self.records if str(r['source']) == str(source)]
            assert len(records) == len(self.config['depths']) * len(self.config['seeds']), f'Incomplete source {source}'
            for partition in ['train', 'val', 'test']:
                assert len({r['sampling'][f'{partition}_sample_ids_digest'] for r in records}) == 1, f'Inputs differ for source {source}'
            assert len({r['dataset_digest'] for r in records}) == 1, f'Dataset digest differs for source {source}'
        for seed in self.config['seeds']:
            records = [r for r in self.records if r['seed'] == seed]
            assert len({r['initial_state_digest'] for r in records}) == 1, f'Initial model differs across sources/depths for seed {seed}'

    def adaptation(self, folder, sources, targets, feature_source, expected_seed):
        metadata = json.loads((folder / 'adaptation_metadata.json').read_text())
        assert metadata['target_labels_used_for_fitting'] is False, 'Target labels used for fitting'
        assert metadata['target_test_features_used_for_fitting'] is False, 'Target test used for fitting'
        assert metadata['feature_source'] == feature_source, 'Incorrect adaptation feature provenance'
        assert metadata['seed'] == expected_seed, 'Incorrect adaptation seed'
        results = pd.read_csv(folder / 'adaptation_results.csv', dtype={'source': str, 'target': str},
                              float_precision='round_trip')
        predictions = _read_csv(folder / 'adaptation_predictions.csv')
        predictions['source'], predictions['target'] = predictions['source'].astype(str), predictions['target'].astype(str)
        expected_pairs = {(str(s), str(t)) for s in sources for t in targets if str(s) != str(t)}
        audits = {(str(p['source']), str(p['target'])): p for p in metadata['pairs']}
        assert len(audits) == len(metadata['pairs']) and set(audits) == expected_pairs, 'Missing/duplicate/unexpected adaptation metadata pair'
        expected_keys = {(s, t, m) for s, t in expected_pairs for m in METHODS}
        result_keys = list(zip(results['source'], results['target'], results['method']))
        assert len(result_keys) == len(set(result_keys)) and set(result_keys) == expected_keys, 'Adaptation result method/pair coverage mismatch'
        assert set(zip(predictions['source'], predictions['target'], predictions['method'])) == expected_keys, 'Adaptation prediction pair coverage mismatch'
        assert results['feature_source'].eq(feature_source).all() and predictions['feature_source'].eq(feature_source).all(), 'Prediction/result feature source mismatch'
        frame = self.frame
        def domain(center, partition):
            return frame.loc[frame['center'].eq(center) & frame['partition'].eq(partition)].reset_index(drop=True)
        for source, target in sorted(expected_pairs):
            pair = audits[(source, target)]
            roles = {'projection_fit_sample_ids': domain(source, 'train'),
                     'source_labeled_fit_sample_ids': domain(source, 'train'),
                     'target_unlabeled_fit_sample_ids': domain(target, 'train'),
                     'selection_sample_ids': domain(source, 'val'),
                     'evaluation_sample_ids': domain(target, 'test')}
            for role, expected in roles.items():
                ids = [str(value) for value in pair[role]]
                assert len(ids) == len(set(ids)) and set(ids) == set(expected['sample_id']), f'{source}->{target} wrong {role} roles/IDs'
            fitted = set(pair['projection_fit_sample_ids']) | set(pair['source_labeled_fit_sample_ids']) | set(pair['target_unlabeled_fit_sample_ids'])
            assert not fitted.intersection(frame.loc[frame['partition'].eq('test'), 'sample_id']), 'Test IDs used for fitting'
            for method in METHODS:
                group = predictions.loc[predictions['source'].eq(source) & predictions['target'].eq(target) & predictions['method'].eq(method)].reset_index(drop=True)
                expected = roles['evaluation_sample_ids']
                _bind_predictions(group, expected, ['center', 'patient', 'slide', 'label', 'partition'])
                assert np.array_equal(group['prediction'].to_numpy(), (group['probability'].to_numpy() >= .5).astype(int)), 'Adaptation hard prediction differs from probability'
                recorded = results.loc[results['source'].eq(source) & results['target'].eq(target) & results['method'].eq(method)].iloc[0]
                computed = _score(group['label'], group['probability'])
                _same_number(recorded['target_test_n'], computed.pop('n'), 'adaptation n')
                computed['balanced_accuracy'] = float(balanced_accuracy_score(group['label'], group['probability'] >= .5))
                computed['log_loss'] = float(log_loss(group['label'], np.column_stack([1-group['probability'], group['probability']]), labels=[0, 1]))
                for name, value in computed.items():
                    _same_number(recorded[name], value, f'{source}->{target} {method} {name}')
                history_key = 'baseline_C_source_val' if method == 'source_only_logistic' else 'coral_C_source_val'
                history = pair[history_key]
                assert history, 'Missing source-val hyperparameter history'
                chosen = max(history, key=lambda r: (r['accuracy'], -r['log_loss'], -r['C']))
                _same_number(recorded['selected_C'], chosen['C'], 'selected adaptation C')
                _same_number(recorded['source_val_accuracy'], chosen['accuracy'], 'selected adaptation validation accuracy')
                _same_number(recorded['source_val_roc_auc'], chosen['roc_auc'], 'selected adaptation validation AUC')
                self.adaptation_methods += 1
                self.adaptation_predictions += len(group)
            self.adaptation_pairs += 1

    def execute(self):
        self.check(self.manifest_path, 'patient/slide-disjoint nonempty manifest', self.manifest)
        expected_runs = list(itertools.product(self.config['sources'], self.config['depths'], self.config['seeds']))
        def matrix_coverage():
            expected = {self.results/'runs'/f'source_{s}'/d/f'seed_{seed}'/'metrics.json' for s,d,seed in expected_runs}
            actual = set((self.results/'runs').glob('source_*/*/seed_*/metrics.json'))
            assert actual == expected, f'Expected {len(expected)} run metrics, found {len(actual)}; missing={len(expected-actual)}, extra={len(actual-expected)}'
        self.check(self.results/'runs', 'exact configured run matrix', matrix_coverage)
        def saved_config():
            assert json.loads((self.results/'config.json').read_text()) == self.config, 'Saved config.json mismatch'
        self.check(self.results/'config.json', 'saved experiment config matches expected config', saved_config)
        if hasattr(self, 'frame'):
            for source, depth, seed in expected_runs:
                path = self.results/'runs'/f'source_{source}'/depth/f'seed_{seed}'
                self.check(path, 'completed run, validation selection and independent test metrics',
                           lambda s=source,d=depth,z=seed: self.run(s,d,z))
            self.check(self.results/'runs', 'same inputs and initial weights across comparison matrix', self.common_inputs_and_initialization)
            if not self.main_only:
                sources = self.config.get('adaptation_sources', self.config['sources'])
                targets = self.config.get('adaptation_targets', self.config.get('targets', [0,1,2,3,4]))
                reference_seed = int(self.config.get('seed', 42))
                self.check(self.results/'analysis', 'reference adaptation predictions, metrics and train-only fitting roles',
                           lambda: self.adaptation(self.results/'analysis', sources, targets, REFERENCE, reference_seed))
                for seed in self.config.get('adaptation_seeds', [self.config['seeds'][0]]):
                    for source in self.config['sources']:
                        for depth in self.config['depths']:
                            folder = self.results/'depth_adaptation'/f'source_{source}'/depth/f'seed_{seed}'
                            feature_source = f'source_{source}/{depth}/seed_{seed}'
                            self.check(folder, 'learned-representation adaptation predictions and fitting roles',
                                       lambda f=folder,s=source,feature=feature_source,z=seed:
                                       self.adaptation(f,[s],targets,feature,z))
        report = {'status': 'passed' if not self.failures else 'failed', 'complete': not self.failures,
                  'expected_runs': len(expected_runs), 'checked_runs': self.runs,
                  'patch_predictions': self.patch_predictions, 'adaptation_pairs': self.adaptation_pairs,
                  'adaptation_methods': self.adaptation_methods, 'adaptation_predictions': self.adaptation_predictions,
                  'manifest': str(self.manifest_path.resolve()), 'results': str(self.results.resolve()),
                  'scope': 'main project only' if self.main_only else 'main project and adaptation',
                  'checkpoint_bytes_downloaded': False, 'checkpoint_bytes_verified_by_this_audit': False,
                  'checkpoint_scope': 'Reload verification is checked from saved engine records only; model checkpoint bytes are not loaded or independently verified here.',
                  'metric_tolerance': 1e-6, 'assertions': self.assertions, 'failures': self.failures}
        self.results.mkdir(parents=True, exist_ok=True)
        (self.results/('audit_main.json' if self.main_only else 'audit.json')).write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        return report


def audit_results(manifest, results, config, main_only=False):
    return Audit(manifest, results, config, main_only=main_only).execute()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='Generated split-specific config.json for the results being audited')
    parser.add_argument('--main-only', action='store_true', help='只核对主项目，写audit_main.json并保留扩展审核记录')
    args = parser.parse_args()
    report = audit_results(args.manifest, args.results, json.loads(args.config.read_text()), main_only=args.main_only)
    print(json.dumps({key: report[key] for key in ['status','expected_runs','checked_runs','patch_predictions','adaptation_pairs','adaptation_predictions','failures']}, indent=2))
    raise SystemExit(0 if report['complete'] else 1)


if __name__ == '__main__':
    main()
