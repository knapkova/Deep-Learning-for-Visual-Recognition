"""微调公平性、冻结 BN、来源隔离、checkpoint 重载与缓存失效回归。"""
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from camelyon.models import DEPTHS, build_model, set_train_mode, trainable_prefixes
from camelyon import train


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(self.layer4(self.layer3(self.layer1(x))).mean((2, 3)))


def tiny_build(depth, seed=42, pretrained=True):
    torch.manual_seed(seed)
    model = TinyNet()
    prefixes = trainable_prefixes(depth)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(prefixes))
    return model


def fixture_loader(root, frame, config, training, seed):
    xs = torch.stack([torch.from_numpy(np.asarray(Image.open(Path(root) / row.path)).copy())
                      .permute(2, 0, 1).float() / 255 for row in frame.itertuples()])
    ys = torch.tensor(frame.label.tolist())
    return DataLoader(TensorDataset(xs, ys), batch_size=4, shuffle=training,
                      generator=torch.Generator().manual_seed(seed))


class ModelTests(unittest.TestCase):
    def test_resnet_depths_have_identical_initial_weights_and_precise_flags(self):
        first = None
        counts = []
        for depth in DEPTHS:
            model = build_model(depth, seed=17, pretrained=False)
            digest = train._state_digest(model)
            if first is None:
                first = digest
            self.assertEqual(first, digest)
            flags = {name: p.requires_grad for name, p in model.named_parameters()}
            for name, value in flags.items():
                self.assertEqual(value, name.startswith(trainable_prefixes(depth)))
            self.assertEqual(model.fc.out_features, 2)
            counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
            del model
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[0], 4098)
        self.assertEqual(len(set(counts)), 4)

    def test_frozen_bn_stats_and_parameters_stay_frozen_during_real_step(self):
        model = tiny_build('layer4')
        old = {name: value.clone() for name, value in model.state_dict().items()}
        set_train_mode(model, 'layer4')
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        loss = nn.functional.cross_entropy(model(torch.randn(4, 3, 8, 8)), torch.tensor([0, 1, 0, 1]))
        loss.backward()
        opt.step()
        for name, value in model.state_dict().items():
            if name.startswith(('layer1.', 'layer3.')):
                self.assertTrue(torch.equal(old[name], value), name)
        self.assertFalse(torch.equal(old['layer4.1.running_mean'], model.layer4[1].running_mean))
        self.assertFalse(torch.equal(old['fc.weight'], model.fc.weight))
        self.assertFalse(model.layer1[1].training)
        self.assertTrue(model.layer4[1].training)


class EngineTests(unittest.TestCase):
    def make_fixture(self, root):
        (root / 'data').mkdir()
        rows = []
        for center in [0, 1]:
            for partition in ['train', 'val', 'test']:
                for label in [0, 1]:
                    for n in range(3):
                        sid = f'{center}-{partition}-{label}-{n}'
                        rel = f'data/{sid}.png'
                        Image.new('RGB', (8, 8), (label * 180 + 20,) * 3).save(root / rel)
                        rows.append(dict(sample_id=sid, path=rel, center=center,
                                         patient=f'{center}-{partition}', slide=f'{center}-{partition}',
                                         label=label, partition=partition))
        pd.DataFrame(rows).to_csv(root / 'data/manifest.csv', index=False)

    def test_source_only_training_reload_and_signature_cache(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            config = dict(epochs=2, batch_size=4, image_size=8, device='cpu',
                          backbone_lr=.001, head_lr=.01, max_train_per_class=2,
                          max_val_per_class=2, max_test_per_class=2)
            seen = []
            def recorded_loader(root, frame, config, training, seed):
                seen.append((training, set(frame.center), set(frame.partition)))
                return fixture_loader(root, frame, config, training, seed)
            with patch('camelyon.train.build_model', side_effect=tiny_build), \
                 patch('camelyon.train._loader', side_effect=recorded_loader):
                result = train.run_experiment(root, config, 0, 'layer4', 7, root / 'output')
                self.assertTrue(result['reload_verified'])
                self.assertEqual(result['train_n'], 4)
                self.assertEqual(result['val_n'], 4)
                self.assertEqual(result['test_n'], 8)
                self.assertEqual(result['selection']['center'], '0')
                self.assertFalse(result['selection']['target_test_used_for_selection'])
                self.assertEqual(result['test_by_center']['0']['role'], 'ID')
                self.assertEqual(result['test_by_center']['1']['role'], 'OOD')
                self.assertEqual(seen[0], (True, {0}, {'train'}))
                self.assertEqual(seen[1], (False, {0}, {'val'}))
                self.assertEqual(seen[-1], (False, {0, 1}, {'test'}))
                pred = pd.read_csv(root / 'output/predictions.csv')
                self.assertEqual(pred.sample_id.tolist(), sorted(pred.sample_id.tolist()))
                self.assertTrue(pred.probability.between(0, 1).all())
                cached = train.run_experiment(root, config, 0, 'layer4', 7, root / 'output')
                self.assertEqual(cached, result)
                self.assertEqual(len(seen), 3)
                # Real image-byte changes invalidate resume even if the manifest is identical.
                image_path = root / f'data/{pred.iloc[0].sample_id}.png'
                Image.new('RGB', (8, 8), (100, 30, 20)).save(image_path)
                with self.assertRaisesRegex(ValueError, 'signature mismatch'):
                    train.run_experiment(root, config, 0, 'layer4', 7, root / 'output')
                updated = train.run_experiment(root, config, 0, 'layer4', 7, root / 'updated_output')
                self.assertNotEqual(updated['signature'], result['signature'])
                self.assertEqual(len(seen), 6)

    def test_epoch_resume_restores_optimizer_rng_loader_and_best_export(self):
        torch.set_num_threads(1)
        class Interrupted(RuntimeError):
            pass
        class RandomTinyNet(TinyNet):
            def forward(self, x):
                if self.training:
                    # 三个随机流均参与梯度，不能只恢复torch或模型权重。
                    x = x + torch.rand_like(x)*.03 + float(np.random.random())*.02 + random.random()*.01
                return super().forward(x)
        def stochastic_build(depth, seed=42, pretrained=True):
            torch.manual_seed(seed)
            model = RandomTinyNet()
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith(trainable_prefixes(depth)))
            return model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            config = dict(epochs=4, batch_size=4, image_size=8, device='cpu',
                          head_lr=.01, backbone_lr=.001, keep_resume_checkpoint=True)
            with patch('camelyon.train.build_model', side_effect=stochastic_build), \
                 patch('camelyon.train._loader', side_effect=fixture_loader):
                uninterrupted = train.run_experiment(root, config, 0, 'full', 17, root/'whole')
                writer = train._save_resume_checkpoint
                def interrupt_after_commit(path, payload):
                    writer(path, payload)
                    if payload['epoch'] == 1:
                        raise Interrupted('simulated process loss after committed epoch, before best export')
                with patch('camelyon.train._save_resume_checkpoint', side_effect=interrupt_after_commit):
                    with self.assertRaises(Interrupted):
                        train.run_experiment(root, config, 0, 'full', 17, root/'resumed')
                state = torch.load(root/'resumed/last.pt', weights_only=False)
                for field in ['optimizer','scaler','rng','loader_generators','history','best_metadata','bad_epochs']:
                    self.assertIn(field, state)
                self.assertEqual(state['epoch'], 1)
                self.assertFalse((root/'resumed/best.pt').exists())
                resumed = train.run_experiment(root, config, 0, 'full', 17, root/'resumed')
                self.assertEqual(resumed['resume']['resumed_from_epoch'], 1)
                self.assertEqual(uninterrupted['checkpoint_state_digest'], resumed['checkpoint_state_digest'])
                last_whole = torch.load(root/'whole/last.pt', weights_only=False)
                last_resumed = torch.load(root/'resumed/last.pt', weights_only=False)
                self.assertEqual(last_whole['state_digest'], last_resumed['state_digest'])
                self.assertEqual(last_whole['bad_epochs'], last_resumed['bad_epochs'])
                columns=['epoch','train_loss','train_accuracy','val_loss','val_accuracy','val_roc_auc']
                pd.testing.assert_frame_equal(pd.read_csv(root/'whole/history.csv')[columns],
                                              pd.read_csv(root/'resumed/history.csv')[columns], check_exact=True)
                pd.testing.assert_frame_equal(pd.read_csv(root/'whole/predictions.csv'),
                                              pd.read_csv(root/'resumed/predictions.csv'), check_exact=True)

    def test_signature_change_and_corrupted_completion_preserve_existing_results(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.make_fixture(root)
            config=dict(epochs=1, device='cpu', image_size=8)
            with patch('camelyon.train.build_model', side_effect=tiny_build), \
                 patch('camelyon.train._loader', side_effect=fixture_loader):
                train.run_experiment(root,config,0,'head',7,root/'output')
                self.assertFalse((root/'output/last.pt').exists())
                original=(root/'output/best.pt').read_bytes()
                for changed in [{'epochs':2},{'head_lr':.1},{'early_stopping_patience':5}]:
                    with self.assertRaisesRegex(ValueError,'signature mismatch'):
                        train.run_experiment(root,{**config,**changed},0,'head',7,root/'output')
                self.assertEqual(original,(root/'output/best.pt').read_bytes())
                prediction_path=root/'output/predictions.csv'
                prediction_path.write_text(prediction_path.read_text()+'corrupted\n')
                with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
                    train.run_experiment(root,config,0,'head',7,root/'output')
                self.assertTrue(prediction_path.read_text().endswith('corrupted\n'))

    def test_early_stop_counter_survives_resume_after_non_improving_epoch(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.make_fixture(root)
            config=dict(epochs=8, early_stopping_patience=2, device='cpu', image_size=8,
                        keep_resume_checkpoint=True)
            evaluator=train._evaluate
            def fixed_validation_loss(model,loader,device):
                y,p,loss=evaluator(model,loader,device)
                return y,p,np.ones_like(loss)
            writer=train._save_resume_checkpoint
            def interrupt(path,payload):
                writer(path,payload)
                if payload['epoch']==2:
                    raise RuntimeError('planned interruption')
            with patch('camelyon.train.build_model',side_effect=tiny_build), \
                 patch('camelyon.train._loader',side_effect=fixture_loader), \
                 patch('camelyon.train._evaluate',side_effect=fixed_validation_loss):
                whole=train.run_experiment(root,config,0,'full',8,root/'whole')
                with patch('camelyon.train._save_resume_checkpoint',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'planned interruption'):
                        train.run_experiment(root,config,0,'full',8,root/'resumed')
                state=torch.load(root/'resumed/last.pt',weights_only=False)
                self.assertEqual(state['bad_epochs'],1)
                result=train.run_experiment(root,config,0,'full',8,root/'resumed')
                self.assertEqual(result['selection']['epochs_completed'],3)
                self.assertEqual(result['selection']['best_epoch'],1)
                self.assertEqual(result['checkpoint_state_digest'],whole['checkpoint_state_digest'])
                self.assertEqual(torch.load(root/'resumed/last.pt',weights_only=False)['bad_epochs'],2)

    def test_recreated_worker_loader_restores_augmentation_and_shuffle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.make_fixture(root)
            frame=train._frame_for(pd.read_csv(root/'data/manifest.csv'),'train',0)
            config=dict(image_size=8,batch_size=4,num_workers=2,augment=True)
            loader=train._loader(root,frame,config,True,51)
            list(loader)  # 完成epoch1，让generator前进。
            saved=train._generator_state(loader)
            expected=list(loader)
            recreated=train._loader(root,frame,config,True,51)
            train._restore_generator(recreated,saved)
            actual=list(recreated)
            self.assertEqual(len(expected),len(actual))
            for (x_expected,y_expected),(x_actual,y_actual) in zip(expected,actual):
                self.assertTrue(torch.equal(x_expected,x_actual))
                self.assertTrue(torch.equal(y_expected,y_actual))

    def test_preverified_image_digest_reuses_hashes_and_rejects_changed_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.make_fixture(root)
            manifest=root/'data/manifest.csv'
            frame=pd.read_csv(manifest)
            index=train.build_image_digest_manifest(root,manifest,'data/image_digests.json')
            # 图像payload不再打开/重新hash；清单自身仍要读取。
            original=Path.open
            image_reads=[]
            def guarded(path,*args,**kwargs):
                if path.suffix=='.png':
                    image_reads.append(str(path))
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',guarded):
                first=train._dataset_digest(root,manifest,[frame],index)
                second=train._dataset_digest(root,manifest,[frame],index)
            self.assertEqual(first,second)
            self.assertEqual(image_reads,[])
            Image.new('RGB',(8,8),(20,60,80)).save(root/frame.iloc[0].path)
            with self.assertRaisesRegex(ValueError,'changed since preverification'):
                train._dataset_digest(root,manifest,[frame],index)

    def test_rebuilt_image_digest_is_stable_after_runtime_file_mtime_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.make_fixture(root)
            manifest=root/'data/manifest.csv'
            frame=pd.read_csv(manifest)
            index=train.build_image_digest_manifest(root,manifest,'data/image_digests.json')
            before=train._dataset_digest(root,manifest,[frame],index)
            for relative in frame.path:
                path=root/relative
                stamp=path.stat().st_mtime_ns+10_000_000_000
                os.utime(path,ns=(stamp,stamp))
            train.build_image_digest_manifest(root,manifest,index)
            after=train._dataset_digest(root,manifest,[frame],index)
            self.assertEqual(before,after)

    def test_legacy_sample_artifacts_without_identity_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.make_fixture(root)
            output=root/'legacy_sample'
            output.mkdir()
            checkpoint=output/'best.pt'
            checkpoint.write_bytes(b'preserved legacy artifact')
            with self.assertRaisesRegex(ValueError,'no compatible identity'):
                train.run_experiment(root,dict(epochs=1,device='cpu'),0,'head',42,output)
            self.assertEqual(checkpoint.read_bytes(),b'preserved legacy artifact')


    def test_patient_partition_leakage_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            path = root / 'data/manifest.csv'
            frame = pd.read_csv(path)
            frame.loc[frame.center.eq(0), 'patient'] = 'leaked_patient'
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'patient crosses'):
                train.run_experiment(root, {'epochs': 1}, 0, 'head', 42, root / 'output')

    def test_require_cuda_rejects_cpu_before_model_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            with patch('camelyon.train.build_model') as constructor:
                with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
                    train.run_experiment(root, {'epochs': 1, 'device': 'cpu', 'require_cuda': True},
                                         0, 'head', 42, root / 'output')
                constructor.assert_not_called()

    def test_public_sampling_is_independent_of_model_depth_and_seed(self):
        frame = pd.DataFrame([dict(sample_id=str(i), center=c, label=y, partition='train')
                              for i, (c, y) in enumerate((c, y) for c in range(2) for y in [0, 1] for _ in range(8))])
        a = train._frame_for(frame, 'train', 0, 3, 42)
        b = train._frame_for(frame.sample(frac=1, random_state=8), 'train', 0, 3, 42)
        self.assertEqual(a.sample_id.tolist(), b.sample_id.tolist())
        self.assertEqual(a.groupby('label').size().tolist(), [3, 3])


if __name__ == '__main__':
    unittest.main()
