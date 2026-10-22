"""按来源医院训练、仅按同医院验证集选点、重载最佳模型后统一测试。"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .models import build_model, seed_everything, set_train_mode

ENGINE_VERSION = 2
_IMAGE_INDEX_CACHE: dict[tuple, dict] = {}


def _plain_config(config: Any) -> dict:
    return dict(config) if isinstance(config, dict) else vars(config).copy()


def _frame_for(frame: pd.DataFrame, partition: str, center: Any = None,
               cap: int | None = None, sample_seed: int = 42) -> pd.DataFrame:
    """固定数据抽样种子，不因微调深度/模型初始化 seed 改变公共输入。"""
    selected = frame.loc[frame.partition.eq(partition)].copy()
    if center is not None:
        selected = selected.loc[selected.center.astype(str).eq(str(center))]
    selected = selected.sort_values('sample_id', kind='stable')
    if cap is not None:
        if int(cap) < 1:
            raise ValueError('Per-class caps must be positive or null')
        pieces = []
        for _, group in selected.groupby(['center', 'label'], sort=True):
            pieces.append(group.sample(n=min(len(group), int(cap)), random_state=sample_seed))
        selected = pd.concat(pieces) if pieces else selected
    return selected.sort_values('sample_id', kind='stable').reset_index(drop=True)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_image_digest_manifest(root, manifest_path, output_path) -> Path:
    """对已准备好的真实图像逐字节验证一次；各run可复用SHA，仅检查size/mtime。

    图像替换或复制到另一文件系统后必须重建。不要把此清单当作下载完成证明。
    """
    root = Path(root).resolve()
    manifest_path, output_path = Path(manifest_path), Path(output_path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not output_path.is_absolute():
        output_path = root / output_path
    records = {}
    paths = sorted(set(pd.read_csv(manifest_path, usecols=['path']).path.astype(str)))
    for number, relative in enumerate(paths, 1):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Image path escapes project root: {relative}')
        before = path.stat()
        checksum = _file_digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Image changed while hashing: {relative}')
        records[relative] = {'size': after.st_size, 'mtime_ns': after.st_mtime_ns,
                             'sha256': checksum}
        if number % 10000 == 0:
            print(f'[image-digests] {number}/{len(paths)} verified', flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _json_write(output_path, {'schema_version': 1, 'source_manifest_sha256': _file_digest(manifest_path),
                              'verification': 'sha256 bytes once; size/mtime checked on each run',
                              'images': records})
    return output_path


def _dataset_digest(root: Path, manifest: Path, frames: list[pd.DataFrame],
                    image_digest_manifest=None) -> str:
    digest = hashlib.sha256(manifest.read_bytes())
    # Image bytes are included: replacing an image invalidates a completed-run cache.
    paths = sorted(set(pd.concat(frames).path.astype(str)))
    index = None
    if image_digest_manifest is not None:
        index_path = Path(image_digest_manifest)
        if not index_path.is_absolute():
            index_path = root / index_path
        status = index_path.stat()
        token = (str(index_path.resolve()), status.st_size, status.st_mtime_ns)
        if token not in _IMAGE_INDEX_CACHE:
            value = json.loads(index_path.read_text())
            if value.get('schema_version') != 1 or not isinstance(value.get('images'), dict):
                raise ValueError('Unsupported image digest manifest')
            _IMAGE_INDEX_CACHE[token] = value['images']
        index = _IMAGE_INDEX_CACHE[token]
        digest.update(b'preverified-image-sha256-v1')
    for relative in paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Image path escapes project root: {relative}')
        digest.update(relative.encode())
        if index is not None:
            record = index.get(relative)
            if record is None:
                raise ValueError(f'Image absent from preverified digest manifest: {relative}')
            status = path.stat()
            if (status.st_size, status.st_mtime_ns) != (record['size'], record['mtime_ns']):
                raise ValueError(f'Image changed since preverification; rebuild digest manifest: {relative}')
            digest.update(record['sha256'].encode())
            digest.update(str(record['size']).encode())
            continue
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
    return digest.hexdigest()


def _state_digest(model: nn.Module) -> str:
    return _weights_digest(model.state_dict())


def _weights_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _loader(root: Path, frame: pd.DataFrame, config: dict, training: bool,
            seed: int) -> DataLoader:
    from .data import PatchDataset
    dataset = PatchDataset(root, frame, image_size=int(config.get('image_size', 96)),
                           augment=training and bool(config.get('augment', False)))
    generator = torch.Generator().manual_seed(seed)
    workers = int(config.get('num_workers', 0))
    if config.get('persistent_workers', False):
        raise ValueError('persistent_workers cannot restore worker augmentation RNG; leave it false')
    options = {'prefetch_factor': int(config.get('prefetch_factor', 2))} if workers else {}
    return DataLoader(dataset, batch_size=int(config.get('batch_size', 32)),
                      shuffle=training, num_workers=workers, generator=generator,
                      worker_init_fn=_seed_worker, persistent_workers=False,
                      pin_memory=bool(config.get('pin_memory', torch.cuda.is_available())), **options)


def _seed_worker(worker_id):
    # torch为worker分配的base_seed来自已保存的DataLoader generator。
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _batch_xy(batch):
    if isinstance(batch, dict):
        return batch['image'], batch['label']
    return batch[0], batch[1]


def _metrics(labels: np.ndarray, probs: np.ndarray, losses: np.ndarray) -> dict:
    return {'n': int(len(labels)), 'roc_auc': float(roc_auc_score(labels, probs))
            if len(np.unique(labels)) == 2 else None,
            'accuracy': float(np.mean((probs >= 0.5) == labels)),
            'loss': float(np.mean(losses)),
            'class_counts': {str(k): int((labels == k).sum()) for k in [0, 1]}}


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels, probs, losses = [], [], []
    for batch in loader:
        x, y = _batch_xy(batch)
        y = y.to(device=device, dtype=torch.long)
        logits = model(x.to(device))
        losses.append(nn.functional.cross_entropy(logits, y, reduction='none').cpu().numpy())
        probs.append(logits.softmax(1)[:, 1].cpu().numpy())
        labels.append(y.cpu().numpy())
    if not labels:
        raise ValueError('Cannot evaluate an empty split')
    return np.concatenate(labels), np.concatenate(probs), np.concatenate(losses)


def _json_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _torch_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def _rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'mps': torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA RNG checkpoint requires a CUDA runtime')
        torch.cuda.set_rng_state_all(state['cuda'])
    if state.get('mps') is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'])


def _generator_state(loader):
    return loader.generator.get_state() if loader.generator is not None else None


def _restore_generator(loader, state):
    if state is not None:
        if loader.generator is None:
            raise RuntimeError('Loader generator missing during resume')
        loader.generator.set_state(state)


def _save_resume_checkpoint(path: Path, payload: dict):
    """唯一epoch提交点；成功替换last.pt后，该epoch才算可恢复。"""
    _torch_write(path, payload)


def _check_identity(output_dir: Path, signature: str):
    known = ['run.json', 'best.pt', 'last.pt', 'metrics.json', 'completion.json',
             'history.csv', 'predictions.csv']
    if not any((output_dir / name).exists() for name in known):
        return
    identity = output_dir / 'run.json'
    if not identity.exists():
        raise ValueError('Existing run has no compatible identity; choose a new output directory; existing results preserved')
    if json.loads(identity.read_text()).get('signature') != signature:
        raise ValueError('Run signature mismatch; choose a new output directory; existing results preserved')


def _completed_cache(output_dir: Path, signature: str):
    completion_path = output_dir / 'completion.json'
    if not completion_path.exists():
        return None
    completion = json.loads(completion_path.read_text())
    expected = {'run.json', 'best.pt', 'history.csv', 'predictions.csv', 'metrics.json'}
    if (completion.get('status') != 'complete' or completion.get('signature') != signature or
            set(completion.get('artifacts', {})) != expected):
        raise RuntimeError('Completed-run identity/artifact inventory mismatch; preserve and inspect results')
    for relative, checksum in completion['artifacts'].items():
        path = output_dir / relative
        if not path.is_file() or _file_digest(path) != checksum:
            raise RuntimeError(f'Completed-run artifact checksum mismatch: {relative}; results preserved')
    saved = json.loads((output_dir / 'metrics.json').read_text())
    checkpoint = torch.load(output_dir / 'best.pt', map_location='cpu', weights_only=False)
    if saved.get('signature') != signature or saved.get('status') != 'complete' or checkpoint.get('signature') != signature:
        raise RuntimeError('Completed-run result/checkpoint signature mismatch')
    if checkpoint.get('state_digest') != saved.get('checkpoint_state_digest'):
        raise RuntimeError('Completed-run checkpoint/result weight identity mismatch')
    return saved


def _restore_best_checkpoint(output_dir: Path, resume: dict):
    """last先提交、best后导出；二者之间中断时从last当前权重恢复best。"""
    expected = resume['best_metadata']
    path = output_dir / 'best.pt'
    if expected['epoch'] == resume['epoch']:
        checkpoint = {**expected, 'state_dict': resume['state_dict']}
        if _weights_digest(checkpoint['state_dict']) != expected['state_digest']:
            raise RuntimeError('Resume best/current weight digest mismatch')
        _torch_write(path, checkpoint)
    elif path.exists():
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if (checkpoint.get('signature') != resume['signature'] or
                checkpoint.get('epoch') != expected['epoch'] or
                _weights_digest(checkpoint['state_dict']) != expected['state_digest']):
            raise RuntimeError('Persisted best checkpoint does not match committed epoch state')
    else:
        raise RuntimeError('Committed earlier best checkpoint is missing; results preserved')


def run_experiment(root, config, source, depth, seed, output_dir) -> dict:
    """同签名run按已提交epoch恢复；不同实验绝不覆盖已有结果。"""
    started = time.perf_counter()
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    config = _plain_config(config)
    seed = int(seed)
    manifest_path = Path(config.get('manifest_path', 'data/manifest.csv'))
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    frame = pd.read_csv(manifest_path, dtype={'sample_id': str})
    required = {'sample_id', 'path', 'center', 'patient', 'slide', 'label', 'partition'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Manifest missing columns: {sorted(required - set(frame.columns))}')
    if frame.sample_id.duplicated().any() or frame.path.duplicated().any():
        raise ValueError('Manifest contains duplicate sample IDs or paths')
    if frame[list(required)].isna().any().any():
        raise ValueError('Manifest metadata must not contain null values')
    if not set(frame.label.unique()).issubset({0, 1}):
        raise ValueError('Labels must be 0/1')
    if not set(frame.partition.unique()).issubset({'train', 'val', 'test'}):
        raise ValueError('Partitions must be train/val/test')
    for group in ['patient', 'slide']:
        if (frame.groupby(['center', group]).partition.nunique() > 1).any():
            raise ValueError(f'{group} crosses train/val/test partitions')
    sample_seed = int(config.get('sampling_seed', 42))
    train = _frame_for(frame, 'train', source, config.get('max_train_per_class'), sample_seed)
    val = _frame_for(frame, 'val', source, config.get('max_val_per_class'), sample_seed)
    test = _frame_for(frame, 'test', None, config.get('max_test_per_class'), sample_seed)
    for name, split in [('source train', train), ('source val', val)]:
        if set(split.label.unique()) != {0, 1}:
            raise ValueError(f'{name} must contain both classes')
    if test.empty:
        raise ValueError('No held-out test samples')
    dataset_digest = _dataset_digest(root, manifest_path, [train, val, test],
                                     config.get('image_digest_manifest'))
    spec = {'config': config, 'source': str(source), 'depth': depth, 'seed': seed,
            'dataset_digest': dataset_digest, 'engine_version': ENGINE_VERSION,
            'torch_version': str(torch.__version__)}
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    metrics_path = output_dir / 'metrics.json'
    checkpoint_path = output_dir / 'best.pt'
    resume_path = output_dir / 'last.pt'
    _check_identity(output_dir, signature)
    saved = _completed_cache(output_dir, signature)
    if saved is not None:
        print(f'[resume] source={source} depth={depth} seed={seed}: verified complete matching run', flush=True)
        return saved
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = config.get('device', 'auto')
    if device_name == 'auto':
        device_name = ('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
    if config.get('require_cuda', False) and (torch.device(device_name).type != 'cuda' or not torch.cuda.is_available()):
        raise RuntimeError('This formal experiment requires CUDA; select a free Colab GPU runtime.')
    device = torch.device(device_name)
    if config.get('persistent_workers', False):
        raise ValueError('persistent_workers cannot restore worker augmentation RNG; leave it false')
    epochs = int(config.get('epochs', 5))
    if epochs < 1:
        raise ValueError('epochs must be >= 1')
    patience = config.get('early_stopping_patience')
    if patience is not None and int(patience) < 1:
        raise ValueError('early_stopping_patience must be positive or null')
    _json_write(output_dir / 'run.json', {'signature': signature, 'spec': spec})
    resume = None
    if resume_path.exists():
        resume = torch.load(resume_path, map_location='cpu', weights_only=False)
        if resume.get('signature') != signature or resume.get('schema_version') != 1:
            raise ValueError('Epoch checkpoint signature/schema mismatch; existing results preserved')
        if resume.get('device_type') != device.type:
            raise ValueError('Epoch checkpoint device type changed; resume on the original device type')
        if _weights_digest(resume['state_dict']) != resume['state_digest']:
            raise RuntimeError('Epoch checkpoint weight digest mismatch')
        if (resume['epoch'] != len(resume['history']) or
                [r['epoch'] for r in resume['history']] != list(range(1, resume['epoch']+1))):
            raise RuntimeError('Epoch checkpoint history is not contiguous')
    elif any((output_dir / name).exists() for name in ['best.pt', 'history.csv', 'metrics.json', 'predictions.csv']):
        raise RuntimeError('Partial artifacts have no committed epoch checkpoint; results preserved')
    seed_everything(seed)
    model = build_model(depth, seed=seed,
                        pretrained=bool(config.get('pretrained', True)) and resume is None).to(device)
    initial_state_digest = resume['initial_state_digest'] if resume else _state_digest(model)
    head, backbone = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (head if name.startswith('fc.') else backbone).append(parameter)
    groups = [{'params': head, 'lr': float(config.get('head_lr', 1e-3)), 'name': 'head'}]
    if backbone:
        groups.append({'params': backbone, 'lr': float(config.get('backbone_lr', 1e-4)), 'name': 'backbone'})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config.get('weight_decay', 1e-4)))
    use_amp = bool(config.get('amp', False)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    train_loader = _loader(root, train, config, True, seed)
    val_loader = _loader(root, val, config, False, seed)
    history, best_loss, best_epoch, bad_epochs = [], float('inf'), 0, 0
    best_metadata = None
    completed_epochs, prior_runtime = 0, 0.0
    if resume is not None:
        model.load_state_dict(resume['state_dict'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        history = resume['history']
        best_loss, best_epoch, bad_epochs = resume['best_loss'], resume['best_epoch'], resume['bad_epochs']
        best_metadata = resume['best_metadata']
        completed_epochs, prior_runtime = resume['epoch'], resume['runtime_seconds']
        _restore_best_checkpoint(output_dir, resume)
        _restore_generator(train_loader, resume['loader_generators']['train'])
        _restore_generator(val_loader, resume['loader_generators']['val'])
        # 必须在model/optimizer/loader重建之后恢复，避免初始化消耗随机流。
        _restore_rng(resume['rng'])
        print(f'[resume] source={source} depth={depth} seed={seed}: epoch {completed_epochs} committed', flush=True)
    stopped = patience is not None and bad_epochs >= int(patience)
    for epoch in range(completed_epochs + 1, epochs + 1) if not stopped else []:
        epoch_started = time.perf_counter()
        set_train_mode(model, depth)
        train_loss, correct, count = 0.0, 0, 0
        for batch in train_loader:
            x, y = _batch_xy(batch)
            x, y = x.to(device, non_blocking=device.type == 'cuda'), y.to(device=device, dtype=torch.long, non_blocking=device.type == 'cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss at epoch {epoch}')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = len(y)
            train_loss += float(loss.detach()) * n
            correct += int((logits.detach().argmax(1) == y).sum())
            count += n
        y_val, p_val, l_val = _evaluate(model, val_loader, device)
        vm = _metrics(y_val, p_val, l_val)
        if not np.isfinite(vm['loss']):
            raise RuntimeError(f'Non-finite source validation loss at epoch {epoch}')
        improved = vm['loss'] < best_loss
        if improved:
            best_loss, best_epoch, bad_epochs = vm['loss'], epoch, 0
            best_metadata = {'epoch': epoch, 'val_loss': best_loss, 'signature': signature,
                             'spec': spec, 'initial_state_digest': initial_state_digest,
                             'state_digest': _state_digest(model)}
        else:
            bad_epochs += 1
        history.append({'epoch': epoch, 'train_loss': train_loss / count, 'train_accuracy': correct / count,
                        'val_loss': vm['loss'], 'val_accuracy': vm['accuracy'], 'val_roc_auc': vm['roc_auc'],
                        'epoch_seconds': time.perf_counter() - epoch_started})
        state = _cpu_state(model.state_dict())
        payload = {'schema_version': 1, 'signature': signature, 'spec': spec,
                   'device_type': device.type,
                   'epoch': epoch, 'state_dict': state, 'state_digest': _weights_digest(state),
                   'optimizer': _cpu_state(optimizer.state_dict()), 'scaler': scaler.state_dict(),
                   'rng': _rng_state(), 'loader_generators': {'train': _generator_state(train_loader),
                                                            'val': _generator_state(val_loader)},
                   'history': history, 'best_loss': best_loss, 'best_epoch': best_epoch,
                   'best_metadata': best_metadata, 'bad_epochs': bad_epochs,
                   'initial_state_digest': initial_state_digest,
                   'runtime_seconds': prior_runtime + time.perf_counter() - started}
        _save_resume_checkpoint(resume_path, payload)
        if improved:
            _torch_write(checkpoint_path, {**best_metadata, 'state_dict': state})
        history_temporary = output_dir / 'history.csv.tmp'
        pd.DataFrame(history).to_csv(history_temporary, index=False)
        history_temporary.replace(output_dir / 'history.csv')
        del payload, state
        print(f'[epoch] source={source} depth={depth} seed={seed} {epoch}/{epochs} '
              f'train_loss={train_loss/count:.4f} val_loss={vm["loss"]:.4f} '
              f'val_auc={vm["roc_auc"]:.4f} best_epoch={best_epoch} '
              f'seconds={history[-1]["epoch_seconds"]:.1f}', flush=True)
        if patience is not None and bad_epochs >= int(patience) and bad_epochs > 0:
            break
    # last提交后、history导出前中断时，按last重建完整历史。
    pd.DataFrame(history).to_csv(output_dir / 'history.csv.tmp', index=False)
    (output_dir / 'history.csv.tmp').replace(output_dir / 'history.csv')
    del optimizer, model, resume
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # A fresh model proves the export consumes persisted best weights, not the final in-memory model.
    model = build_model(depth, seed=seed, pretrained=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    if _state_digest(model) != checkpoint['state_digest']:
        raise RuntimeError('Reloaded checkpoint weight digest mismatch')
    model = model.to(device)
    y_val, p_val, l_val = _evaluate(model, val_loader, device)
    reloaded_val = _metrics(y_val, p_val, l_val)
    if not np.isclose(reloaded_val['loss'], best_loss, rtol=1e-4, atol=1e-5):
        raise RuntimeError('Reloaded checkpoint validation loss differs from selected best epoch')
    test_loader = _loader(root, test, config, False, seed)
    y_test, p_test, l_test = _evaluate(model, test_loader, device)
    predictions = test[['sample_id', 'center', 'label', 'partition']].copy()
    if not np.array_equal(y_test, predictions.label.to_numpy()):
        raise RuntimeError('Stable prediction order/label correspondence failed')
    predictions['probability'] = p_test
    predictions['source'] = source
    predictions.to_csv(output_dir / 'predictions.csv', index=False)
    per_center = {}
    for center in sorted(test.center.unique(), key=str):
        mask = test.center.eq(center).to_numpy()
        per_center[str(center)] = {**_metrics(y_test[mask], p_test[mask], l_test[mask]),
                                   'role': 'ID' if str(center) == str(source) else 'OOD'}
    result = {**spec, 'signature': signature, 'status': 'complete',
              'train_n': len(train), 'val_n': len(val), 'test_n': len(test),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'total_parameters': sum(p.numel() for p in model.parameters()),
              'initial_state_digest': initial_state_digest,
              'checkpoint': 'best.pt', 'checkpoint_state_digest': checkpoint['state_digest'],
              'model_spec': {'architecture': 'resnet50', 'classes': ['non-tumor', 'tumor'],
                             'weights': 'IMAGENET1K_V2' if config.get('pretrained', True) else None,
                             'head_initialization_seed': seed,
                             'frozen_batchnorm': 'eval; running statistics and affine parameters frozen'},
              'selection': {'split': 'source_val', 'center': str(source),
                            'criterion': 'minimum cross-entropy loss', 'best_epoch': best_epoch,
                            'best_val_loss': best_loss, 'epochs_completed': len(history),
                            'target_test_used_for_selection': False},
              'reloaded_validation': reloaded_val, 'reload_verified': True,
              'test_by_center': per_center, 'runtime_seconds': prior_runtime + time.perf_counter() - started,
              'resume': {'epoch_boundary_only': True, 'resumed_from_epoch': completed_epochs,
                         'checkpoint_schema': 1},
              'environment': {'torch': torch.__version__, 'device': str(device),
                              'platform': platform.platform(), 'amp': use_amp,
                              'cuda_version': torch.version.cuda, 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                              'determinism': 'seeded deterministic algorithms with warn_only=True; MPS best effort'},
              'sampling': {'seed': sample_seed, 'same_inputs_across_depths': True,
                           'train_sample_ids_digest': hashlib.sha256('\n'.join(train.sample_id).encode()).hexdigest(),
                           'val_sample_ids_digest': hashlib.sha256('\n'.join(val.sample_id).encode()).hexdigest(),
                           'test_sample_ids_digest': hashlib.sha256('\n'.join(test.sample_id).encode()).hexdigest()}}
    _json_write(metrics_path, result)
    artifacts = ['run.json', 'best.pt', 'history.csv', 'predictions.csv', 'metrics.json']
    _json_write(output_dir / 'completion.json', {'signature': signature, 'status': 'complete',
                'artifacts': {name: _file_digest(output_dir / name) for name in artifacts}})
    if not config.get('keep_resume_checkpoint', False):
        # 只清除本run已完成的临时optimizer快照，完整best与验收产物保留。
        resume_path.unlink(missing_ok=True)
    print(f'[complete] source={source} depth={depth} seed={seed} best_epoch={best_epoch} '
          f'runtime={result["runtime_seconds"]:.1f}s', flush=True)
    return result
