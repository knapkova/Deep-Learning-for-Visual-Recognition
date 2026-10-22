"""固定参考表征、逐通道一维 W1 与不接触目标测试集的 CORAL 扩展。

References (primary sources):
* Stacke et al., Measuring Domain Shift for Deep Learning in Histopathology,
  JBHI 2021. https://doi.org/10.1109/JBHI.2020.3032060
* Sun, Feng & Saenko, Return of Frustratingly Easy Domain Adaptation,
  AAAI 2016. https://doi.org/10.1609/aaai.v30i1.10306
* Torchvision ResNet50_Weights.IMAGENET1K_V2:
  https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html

这里的 shift 是 avgpool 的 2048 个边际分布的一维 Wasserstein-1 距离均值，
并非多元 Wasserstein-2，也不是 OT 域适应。类条件 shift 使用 train 标签进行
事后分析；这些目标标签绝不传入无监督适应的拟合。CORAL 是固定表征上的
二阶统计对齐，不是端到端 Deep CORAL；分类器重新拟合为 logistic regression。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


META_FIELDS = ("sample_id", "center", "patient", "slide", "label", "partition")
REFERENCE_ID = "torchvision.resnet50.IMAGENET1K_V2.avgpool2048"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _manifest(root, config):
    path = Path(config.get("manifest_path", "data/manifest.csv"))
    if not path.is_absolute():
        path = Path(root) / path
    frame = pd.read_csv(path, dtype={key: str for key in META_FIELDS if key != "label"})
    missing = set(META_FIELDS + ("path",)) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    _validate_arrays(np.zeros((len(frame), 1)), {key: frame[key].to_numpy() for key in META_FIELDS})
    return path, frame


def _validate_arrays(features, arrays):
    if features.ndim != 2 or len(features) == 0 or features.shape[1] == 0:
        raise ValueError("Features must be a nonempty N x D array")
    if not np.isfinite(features).all():
        raise ValueError("Features contain nonfinite values")
    for key in META_FIELDS:
        if key not in arrays or arrays[key].ndim != 1 or len(arrays[key]) != len(features):
            raise ValueError(f"Invalid or missing metadata: {key}")
        if pd.isna(arrays[key]).any():
            raise ValueError(f"Null metadata: {key}")
    if len(np.unique(arrays["sample_id"].astype(str))) != len(features):
        raise ValueError("Duplicate sample_id")
    if not set(np.unique(arrays["label"])).issubset({0, 1}):
        raise ValueError("Only binary labels 0/1 are supported")
    if not set(np.unique(arrays["partition"])).issubset({"train", "val", "test"}):
        raise ValueError("Unknown partition; expected train/val/test")
    frame = pd.DataFrame(arrays)
    for group in ("patient", "slide"):
        if (frame.groupby(["center", group])["partition"].nunique() > 1).any():
            raise ValueError(f"{group} crosses train/val/test partitions")


def _load_features(path):
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float64)
        arrays = {key: archive[key] for key in META_FIELDS}
    arrays["center"] = arrays["center"].astype(str)
    arrays["partition"] = arrays["partition"].astype(str)
    _validate_arrays(features, arrays)
    arrays["label"] = arrays["label"].astype(int)
    return features, arrays


def _extract_model_features(root, frame, config, model):
    """共享纯推理路径；调用者负责固定权重与输出 provenance。"""
    import torch
    from torch.utils.data import DataLoader
    from .data import PatchDataset
    requested = config.get("device", "auto")
    device = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    model.fc = torch.nn.Identity()
    model.requires_grad_(False).eval().to(device)
    loader = DataLoader(PatchDataset(root, frame, image_size=int(config.get("image_size", 96)), augment=False),
                        batch_size=int(config.get("feature_batch_size", config.get("reference_batch_size", 64))),
                        shuffle=False, num_workers=0)
    chunks = []
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            images = batch.get("image", batch.get("images")) if isinstance(batch, dict) else batch[0] if isinstance(batch, (tuple, list)) else batch
            values = model(images.to(device)).cpu().numpy().astype(np.float32)
            if values.ndim != 2 or values.shape[1] != 2048:
                raise ValueError(f"Unexpected ResNet avgpool feature shape: {values.shape}")
            chunks.append(values)
            if step % 20 == 0:
                print(f"Features: {min((step + 1) * loader.batch_size, len(frame))}/{len(frame)}", flush=True)
    return np.concatenate(chunks), str(device)


def _save_feature_archive(path, features, frame):
    arrays = {key: np.asarray(frame[key].to_list(), dtype=np.int64 if key == "label" else str) for key in META_FIELDS}
    _validate_arrays(features, arrays)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, features=features, **arrays)
    os.replace(tmp, path)


def extract_reference_features(root, config, output_dir):
    """提取/复用不可覆盖的 V2 avgpool2048 缓存。返回含路径的 dict。

    config: manifest_path, image_size(96), reference_batch_size(64),
    device(auto/mps/cpu/cuda)。图像由 data.PatchDataset 的无增强实现读取。
    与训练档位无关，始终重新构造 ImageNet V2 模型，绝不读取训练 checkpoint。
    缓存复用要求 manifest、预处理规格和 NPZ 内容指纹均匹配；不匹配时拒绝覆盖。
    """
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, frame = _manifest(root, config)
    feature_path = output_dir / "reference_features.npz"
    metadata_path = output_dir / "reference_features.metadata.json"
    image_size = int(config.get("image_size", 96))
    identity = {"feature_source": REFERENCE_ID, "manifest_sha256": _sha256(manifest_path),
                "image_size": image_size, "preprocessing": "PatchDataset augment=False; ImageNet normalization"}
    if feature_path.exists() or metadata_path.exists():
        if not (feature_path.exists() and metadata_path.exists()):
            raise ValueError("Incomplete feature cache; inspect/remove incomplete files explicitly")
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise ValueError("Immutable reference cache identity mismatch; choose a new output directory")
        if metadata.get("feature_sha256") != _sha256(feature_path):
            raise ValueError("Reference cache checksum mismatch")
        _load_features(feature_path)
        return {"feature_path": str(feature_path), "metadata_path": str(metadata_path), "reused": True}

    import torch
    import torchvision
    from torchvision.models import ResNet50_Weights, resnet50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    features, device = _extract_model_features(root, frame, config, model)
    _save_feature_archive(feature_path, features, frame)
    metadata = {**identity, "feature_sha256": _sha256(feature_path), "rows": len(features),
                "dimensions": features.shape[1], "device": str(device), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "weights_url": ResNet50_Weights.IMAGENET1K_V2.url,
                "scope": "fixed reference only; no fine-tuned checkpoints; no feature-fitting on any partition"}
    _json(metadata_path, metadata)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return {"feature_path": str(feature_path), "metadata_path": str(metadata_path), "reused": False}


def extract_checkpoint_features(root, config, checkpoint_path, output_dir, feature_source):
    """固定某次选定 checkpoint，提取所有分区的 avgpool；只用于深度适应比较。

    使用和参考模型完全相同的无增强提取路径。每个 checkpoint 单独缓存，
    身份绑定文件 SHA256；禁止拿这些特征重新定义主实验的医院 shift 尺子。
    先严格加载完整分类模型的 state_dict，再移除 fc。返回路径 dict。
    """
    import torch
    from .models import build_model
    root, output_dir, checkpoint_path = Path(root), Path(output_dir), Path(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, frame = _manifest(root, config)
    feature_path = output_dir / "checkpoint_features.npz"
    metadata_path = output_dir / "checkpoint_features.metadata.json"
    identity = {"feature_source": str(feature_source), "manifest_sha256": _sha256(manifest_path),
        "checkpoint_sha256": _sha256(checkpoint_path), "image_size": int(config.get("image_size", 96)),
        "preprocessing": "PatchDataset augment=False; ImageNet normalization"}
    if feature_path.exists() or metadata_path.exists():
        if not (feature_path.exists() and metadata_path.exists()):
            raise ValueError("Incomplete checkpoint feature cache")
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise ValueError("Immutable checkpoint cache identity mismatch; choose a new output directory")
        if metadata.get("feature_sha256") != _sha256(feature_path):
            raise ValueError("Checkpoint feature checksum mismatch")
        _load_features(feature_path)
        return {"feature_path": str(feature_path), "metadata_path": str(metadata_path), "reused": True}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = checkpoint.get("spec", {})
    depth = spec.get("depth", config.get("depth"))
    if depth is None:
        raise ValueError("Checkpoint spec.depth or config.depth required")
    model = build_model(depth, seed=int(spec.get("seed", config.get("seed", 42))), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    features, device = _extract_model_features(root, frame, config, model)
    _save_feature_archive(feature_path, features, frame)
    metadata = {**identity, "feature_sha256": _sha256(feature_path), "rows": len(features), "dimensions": 2048,
        "source": spec.get("source"), "depth": depth, "seed": spec.get("seed"), "checkpoint_epoch": checkpoint.get("epoch"),
        "device": device, "scope": "fixed trained representation for post-hoc adaptation; NEVER main shift reference"}
    _json(metadata_path, metadata)
    del model, checkpoint
    if device == "mps":
        torch.mps.empty_cache()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"feature_path": str(feature_path), "metadata_path": str(metadata_path), "reused": False}


def marginal_wasserstein(source, target):
    """Mean of per-feature 1D W1 (not multivariate OT); finite nonempty N×D inputs."""
    source, target = np.asarray(source), np.asarray(target)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Feature matrices must have matching dimensions")
    if not len(source) or not len(target) or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Nonempty finite feature matrices required")
    if len(source) == len(target):
        return float(np.abs(np.sort(source, axis=0) - np.sort(target, axis=0)).mean())
    return float(np.mean([wasserstein_distance(source[:, channel], target[:, channel]) for channel in range(source.shape[1])]))


def compute_shift(feature_path, output_dir):
    """所有有序医院对，train 分区：raw、等类采样、两类条件 W1 及其均值。

    使用每中心每类相同样本数（所有中心最小计数，上限512），固定seed=42。
    目标 train 标签只为这项事后描述分析；不传入模型拟合。返回 CSV 路径。
    """
    provenance_path = Path(feature_path).with_suffix(".metadata.json")
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("feature_source") != REFERENCE_ID:
            raise ValueError("Shift must use the fixed ImageNet reference, never a trained checkpoint")
    features, arrays = _load_features(feature_path)
    centers = sorted(np.unique(arrays["center"]))
    train = arrays["partition"] == "train"
    raw = {center: np.flatnonzero(train & (arrays["center"] == center)) for center in centers}
    class_indices = {(center, label): np.flatnonzero(train & (arrays["center"] == center) & (arrays["label"] == label)) for center in centers for label in (0, 1)}
    per_class = min(512, min(map(len, class_indices.values())))
    if per_class < 1:
        raise ValueError("Every center needs both classes in train for balanced shift analysis")
    rng = np.random.default_rng(42)
    balanced = {key: rng.choice(indices, per_class, replace=False) for key, indices in class_indices.items()}
    rows = []
    for i, source in enumerate(centers):
        for target in centers[i + 1:]:
            left = np.concatenate([balanced[source, label] for label in (0, 1)])
            right = np.concatenate([balanced[target, label] for label in (0, 1)])
            conditional = [marginal_wasserstein(features[balanced[source, label]], features[balanced[target, label]]) for label in (0, 1)]
            common = {"shift_raw": marginal_wasserstein(features[raw[source]], features[raw[target]]),
                      "shift_balanced": marginal_wasserstein(features[left], features[right]),
                      "shift_class0": conditional[0], "shift_class1": conditional[1],
                      "shift_classconditional": float(np.mean(conditional)), "samples_per_class": per_class,
                      "dimensions": features.shape[1], "partition": "train", "seed": 42,
                      "metric": "mean_marginal_1d_wasserstein1"}
            for a, b in ((source, target), (target, source)):
                rows.append({"source": a, "target": b, "source_train_n": len(raw[a]), "target_train_n": len(raw[b]), **common})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "representation_shift.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    _json(output_dir / "representation_shift.metadata.json", {
        "feature_path": str(Path(feature_path).resolve()), "feature_sha256": _sha256(feature_path),
        "metric": "average of marginal 1D W1 distances; NOT multivariate W2",
        "analysis_labels": "target train labels used ONLY for class-conditional shift description, never adaptation",
        "partition": "train", "samples_per_class": per_class, "seed": 42,
        "uncertainty": "patches are clustered by slide/patient; distances are descriptive, not independent-patch significance tests"})
    return path


def coral_transform(source_train, target_train, regularization=1.0):
    """返回 source→target 的 CORAL 矩阵和均值，仅接受无标签 train 特征。

    A=(Cs+λI)^(-1/2)(Ct+λI)^(1/2). 源样本映射为 (x-ms)A+mt；
    目标样本不再变换。均值匹配是显式附加的一阶对齐，λ固定而非目标标签调参。
    """
    source_train, target_train = np.asarray(source_train), np.asarray(target_train)
    if source_train.ndim != 2 or target_train.ndim != 2 or source_train.shape[1] != target_train.shape[1]:
        raise ValueError("CORAL feature dimensions must match")
    if min(len(source_train), len(target_train)) < 2 or regularization <= 0:
        raise ValueError("CORAL needs >=2 train rows per domain and positive regularization")
    dimension = source_train.shape[1]
    cs = np.atleast_2d(np.cov(source_train, rowvar=False)) + regularization * np.eye(dimension)
    ct = np.atleast_2d(np.cov(target_train, rowvar=False)) + regularization * np.eye(dimension)
    def power(matrix, exponent):
        values, vectors = np.linalg.eigh(matrix)
        return (vectors * np.maximum(values, 1e-12) ** exponent) @ vectors.T
    return power(cs, -0.5) @ power(ct, 0.5), source_train.mean(axis=0), target_train.mean(axis=0)


def _metrics(labels, probabilities):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
    labels = np.asarray(labels)
    result = {"accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
              "balanced_accuracy": float(balanced_accuracy_score(labels, probabilities >= 0.5)),
              "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
              "log_loss": float(log_loss(labels, np.column_stack([1 - probabilities, probabilities]), labels=[0, 1]))}
    return result


def _select_classifier(train_x, train_y, val_x, val_y, candidates, seed):
    from sklearn.linear_model import LogisticRegression
    best, history, best_key = None, [], None
    for value in sorted(set(float(x) for x in candidates)):
        if value <= 0:
            raise ValueError("All C candidates must be positive")
        model = LogisticRegression(C=value, solver="lbfgs", max_iter=2000, random_state=seed)
        model.fit(train_x, train_y)
        metrics = _metrics(val_y, model.predict_proba(val_x)[:, 1])
        history.append({"C": value, **metrics})
        # Source validation only; break ties by loss, then stronger regularization.
        key = (metrics["accuracy"], -metrics["log_loss"], -value)
        if best_key is None or key > best_key:
            best, best_key = model, key
    if best is None:
        raise ValueError("C candidates must not be empty")
    return best, history


def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def predict_adaptation_model(model_path, features):
    """加载本项目生成的可信 joblib；对目标空间特征返回阳性概率。"""
    import joblib
    bundle = joblib.load(model_path)
    projected = bundle["scaler"].transform(bundle["pca"].transform(features))
    return bundle["classifier"].predict_proba(projected)[:, 1]


def run_adaptation(feature_path, output_dir, config):
    """固定表征上的配对 source-only logistic vs CORAL+logistic，返回路径 dict。

    支持任何固定 backbone 的同一 NPZ 契约：feature_source 显式记录其来源；
    sources/targets 限定医院对。PCA 和 StandardScaler 只 fit source TRAIN；
    CORAL 只看 source TRAIN 与 target UNLABELED TRAIN。两种方法用同一 C
    候选集、同一 source VAL 规则选模型；target TEST 仅最终预测与打分。
    config: sources, targets, feature_source, pca_components(128),
    C_candidates([.01,.1,1,10]), coral_regularization(1), seed(42)。
    它验证已学固定表征的 post-hoc 适应，不是 nn.Linear 原头或 DeepCORAL。
    """
    import joblib
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    features, arrays = _load_features(feature_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    centers = sorted(np.unique(arrays["center"]))
    sources = [str(x) for x in config.get("sources", centers)]
    targets = [str(x) for x in config.get("targets", centers)]
    if (set(sources) | set(targets)) - set(centers):
        raise ValueError("Requested source/target center absent from feature archive")
    feature_source = str(config.get("feature_source", REFERENCE_ID))
    seed = int(config.get("seed", 42))
    candidates = config.get("C_candidates", config.get("c_candidates", [0.01, 0.1, 1.0, 10.0]))
    regularization = float(config.get("coral_regularization", 1.0))
    rows, prediction_rows, audit = [], [], []
    for source in sources:
        source_train = np.flatnonzero((arrays["center"] == source) & (arrays["partition"] == "train"))
        source_val = np.flatnonzero((arrays["center"] == source) & (arrays["partition"] == "val"))
        if len(source_train) < 3 or len(source_val) < 1 or len(np.unique(arrays["label"][source_train])) != 2:
            raise ValueError(f"Source {source} needs >=3 train rows, both classes and a validation partition")
        components = min(int(config.get("pca_components", config.get("adaptation_pca_components", 128))), features.shape[1], len(source_train) - 1)
        if components < 1:
            raise ValueError("pca_components must be positive")
        solver = "randomized" if components < min(features.shape[1], len(source_train)) else "full"
        pca = PCA(n_components=components, svd_solver=solver, random_state=seed).fit(features[source_train])
        scaler = StandardScaler().fit(pca.transform(features[source_train]))
        st = scaler.transform(pca.transform(features[source_train]))
        sv = scaler.transform(pca.transform(features[source_val]))
        baseline, baseline_history = _select_classifier(st, arrays["label"][source_train], sv, arrays["label"][source_val], candidates, seed)
        for target in targets:
            if source == target:
                continue
            target_train = np.flatnonzero((arrays["center"] == target) & (arrays["partition"] == "train"))
            target_test = np.flatnonzero((arrays["center"] == target) & (arrays["partition"] == "test"))
            if len(target_train) < 2 or len(target_test) < 1:
                raise ValueError(f"Target {target} needs >=2 unlabeled train rows and a test partition")
            # Do not read labels from target_train: target labels are never a fitting input.
            tt = scaler.transform(pca.transform(features[target_train]))
            transform, source_mean, target_mean = coral_transform(st, tt, regularization)
            adapted, adapted_history = _select_classifier((st - source_mean) @ transform + target_mean,
                arrays["label"][source_train], (sv - source_mean) @ transform + target_mean,
                arrays["label"][source_val], candidates, seed)
            pair = f"{_safe(feature_source)}__source{_safe(source)}_target{_safe(target)}"
            for method, classifier, history in (("source_only_logistic", baseline, baseline_history), ("coral_logistic", adapted, adapted_history)):
                model_path = model_dir / f"{pair}__{method}.joblib"
                bundle = {"pca": pca, "scaler": scaler, "classifier": classifier, "method": method,
                          "source": source, "target": target, "feature_source": feature_source,
                          "coral_transform": transform if method == "coral_logistic" else None,
                          "source_mean": source_mean, "target_mean": target_mean,
                          "input_contract": "unmodified features from SAME fixed backbone; target-space inference"}
                joblib.dump(bundle, model_path)
                probabilities = predict_adaptation_model(model_path, features[target_test])
                selected_val = next(item for item in history if item["C"] == classifier.C)
                rows.append({"feature_source": feature_source, "source": source, "target": target, "method": method,
                             "selected_C": float(classifier.C), "source_val_accuracy": selected_val["accuracy"],
                             "source_val_roc_auc": selected_val["roc_auc"], "pca_components": components,
                             "target_test_n": len(target_test), **_metrics(arrays["label"][target_test], probabilities),
                             "model_path": str(model_path)})
                for index, probability in zip(target_test, probabilities):
                    prediction_rows.append({"feature_source": feature_source, "source": source, "target": target,
                        "method": method, **{key: str(arrays[key][index]) for key in META_FIELDS},
                        "probability": float(probability), "prediction": int(probability >= 0.5)})
            audit.append({"source": source, "target": target,
                "projection_fit_sample_ids": arrays["sample_id"][source_train].astype(str).tolist(),
                "source_labeled_fit_sample_ids": arrays["sample_id"][source_train].astype(str).tolist(),
                "target_unlabeled_fit_sample_ids": arrays["sample_id"][target_train].astype(str).tolist(),
                "selection_sample_ids": arrays["sample_id"][source_val].astype(str).tolist(),
                "evaluation_sample_ids": arrays["sample_id"][target_test].astype(str).tolist(),
                "baseline_C_source_val": baseline_history, "coral_C_source_val": adapted_history})
    if not rows:
        raise ValueError("No distinct source-target pairs selected")
    csv_path = output_dir / "adaptation_results.csv"
    predictions_path = output_dir / "adaptation_predictions.csv"
    metadata_path = output_dir / "adaptation_metadata.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    _json(metadata_path, {"feature_path": str(Path(feature_path).resolve()), "feature_sha256": _sha256(feature_path),
        "feature_source": feature_source, "method": "PCA(source train) + source StandardScaler + regularized CORAL covariance and mean alignment + logistic regression",
        "reference": "https://doi.org/10.1609/aaai.v30i1.10306", "coral_regularization": regularization,
        "C_candidates": [float(value) for value in candidates], "selection_rule": "source val accuracy; tie-break logloss then smaller C",
        "scope": "post-hoc fixed-representation adaptation; refit logistic baseline; NOT end-to-end DeepCORAL or original neural classifier",
        "target_labels_used_for_fitting": False, "target_test_features_used_for_fitting": False,
        "seed": seed, "pairs": audit})
    return {"results_path": str(csv_path), "predictions_path": str(predictions_path), "metadata_path": str(metadata_path), "model_dir": str(model_dir)}
