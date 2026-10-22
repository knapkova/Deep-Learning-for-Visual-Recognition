"""真实 Camelyon17 数据缓存与自定义、按病人隔离的五医院实验协议。

这不是 WILDS 官方 train/val/test：每家医院内部先按病人分组，再固定划分
train/val/test。目标医院的 train 只允许作为无标签适应数据使用。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

REPOSITORY = "wltjr1007/Camelyon17-WILDS"
PINNED_REVISION = "d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f"
META_COLUMNS = ["label", "center", "image_id", "patient", "node", "x_coord", "y_coord", "slide"]
MANIFEST_COLUMNS = ["sample_id", "path", "center", "patient", "slide", "label", "partition"]
EXPECTED_FULL_IMAGES = 455954


def _local_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"路径越界: {relative}")
    return path


def _validate_revision(revision):
    if re.fullmatch(r"[0-9a-f]{40}", str(revision)) is None:
        raise ValueError("revision 必须是40位小写commit SHA，不能包含路径。")


def _validate_full_export(metadata, expected_images=EXPECTED_FULL_IMAGES):
    if len(metadata) != expected_images:
        raise ValueError(f"全量metadata应为{expected_images}张，实际{len(metadata)}张；拒绝把部分数据冒称全量。")
    if set(metadata.center.unique()) != set(range(5)):
        raise ValueError("全量metadata必须包含全部5家医院。")
    if not set(metadata.label.unique()).issubset({0, 1}):
        raise ValueError("metadata标签必须为0/1。")
    if metadata.duplicated(["center", "patient", "node", "x_coord", "y_coord"]).any():
        raise ValueError("全量metadata存在重复物理patch。")


def partition_review(partitioned, seed):
    """预先指定seed的覆盖核验；报告缺类，不通过标签/结果重新挑划分。"""
    groups, missing = [], []
    for center in sorted(partitioned.center.unique()):
        for name in ["train", "val", "test"]:
            group = partitioned.loc[partitioned.center.eq(center) & partitioned.partition.eq(name)]
            counts = {str(label): int(group.label.eq(label).sum()) for label in [0, 1]}
            record = {"center": int(center), "partition": name, "images": len(group),
                      "patients": int(group.patient.nunique()), "slides": int(group.slide.nunique()),
                      "class_counts": counts}
            groups.append(record)
            if not all(counts.values()):
                missing.append(record)
    patient_disjoint = not (partitioned.groupby(["center", "patient"]).partition.nunique() > 1).any()
    slide_disjoint = not (partitioned.groupby(["center", "slide"]).partition.nunique() > 1).any()
    return {"partition_seed": int(seed), "total_images": len(partitioned),
            "patient_disjoint": bool(patient_disjoint), "slide_disjoint": bool(slide_disjoint),
            "all_partitions_have_both_classes": not missing, "missing_class_groups": missing,
            "ready": bool(patient_disjoint and slide_disjoint and not missing), "groups": groups,
            "selection_rule": "fixed patient permutation; seeds declared before evaluation; no retry or seed selection based on labels/results"}


def _get(url, *, headers=None, params=None, stream=False):
    for attempt in range(5):
        try:
            response = requests.get(url, headers=headers, params=params, stream=stream,
                                    timeout=(20, 120))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))


def _url(revision, shard):
    return f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{shard}"


class _RangeFile(io.RawIOBase):
    """Parquet 随机读取接口；只预取小 metadata 列，避免下载图像列。"""
    def __init__(self, url):
        response = _get(url, headers={"Range": "bytes=-8"})
        if response.status_code != 206:
            raise RuntimeError("数据源未支持 HTTP Range；拒绝意外下载整个数据集。")
        # 只经过 HF resolver 一次；每组小 Range 重复 resolve 会触发其请求配额。
        self.url = response.url
        self.size = int(response.headers["Content-Range"].split("/")[-1])
        self.position = 0
        self.blocks = [(self.size - 8, response.content)]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def fetch(self, start, size):
        for existing_start, data in self.blocks:
            if existing_start <= start and start + size <= existing_start + len(data):
                return data[start-existing_start:start-existing_start+size]
        response = _get(self.url, headers={"Range": f"bytes={start}-{start+size-1}"})
        if response.status_code != 206 or len(response.content) != size:
            raise RuntimeError("HTTP Range 返回长度不符，不能安全解码 Parquet。")
        self.blocks.append((start, response.content))
        return response.content

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        if size <= 0:
            return b""
        result = self.fetch(self.position, size)
        self.position += size
        return result


def _scan_shard(root, revision, shard, pool):
    cache = root / "data" / "metadata_shards" / revision / Path(shard).name
    if cache.exists():
        return pd.read_parquet(cache)
    local = root / "data" / "raw" / revision / Path(shard).name
    remote = None
    if local.exists():
        parquet = pq.ParquetFile(local)
    else:
        remote = _RangeFile(_url(revision, shard))
        parquet = pq.ParquetFile(remote)
        futures = []
        for i in range(parquet.num_row_groups):
            group = parquet.metadata.row_group(i)
            columns = [group.column(j) for j in range(group.num_columns)
                       if group.column(j).path_in_schema in META_COLUMNS]
            starts = [min(c.data_page_offset, c.dictionary_page_offset or c.data_page_offset) for c in columns]
            start = min(starts)
            end = max(s + c.total_compressed_size for s, c in zip(starts, columns))
            futures.append(pool.submit(remote.fetch, start, end-start))
        for future in futures:
            future.result()
    frame = parquet.read(columns=META_COLUMNS, use_threads=False).to_pandas()
    frame["_shard"] = shard
    frame["_row_in_shard"] = np.arange(len(frame))
    frame["_official_split"] = Path(shard).name.split("-")[0]
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def scan_metadata(root, revision=None):
    """读取全部真实 metadata；网络路径只投影 metadata，不获取完整图像列。"""
    root = Path(root)
    revision = revision or PINNED_REVISION
    directory = root / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"metadata-{revision}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    info = _get(f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{revision}").json()
    if info["sha"] != revision:
        raise ValueError("revision 必须是完整、不可变的 commit SHA。")
    shards = sorted(x["rfilename"] for x in info["siblings"] if x["rfilename"].endswith(".parquet"))
    # 8 个分片调度器共享 24 个小 Range 请求，避免线程相互等待。
    with ThreadPoolExecutor(max_workers=24) as ranges, ThreadPoolExecutor(max_workers=8) as files:
        futures = {files.submit(_scan_shard, root, revision, shard, ranges): shard for shard in shards}
        parts = {}
        for future in as_completed(futures):
            shard = futures[future]
            parts[shard] = future.result()
            print(f"metadata {len(parts)}/{len(shards)}: {shard}", flush=True)
    ordered = [parts[s] for s in shards]
    offsets = {}
    for frame in ordered:
        split = frame["_official_split"].iloc[0]
        frame["_split_row"] = np.arange(len(frame)) + offsets.get(split, 0)
        offsets[split] = offsets.get(split, 0) + len(frame)
    result = pd.concat(ordered, ignore_index=True)
    if result.duplicated(["center", "patient", "node", "x_coord", "y_coord"]).any():
        raise ValueError("发现重复物理 patch，先核对来源。")
    result.to_parquet(path, index=False)
    return result


def assign_partitions(metadata, seed=42):
    """医院内按病人约60/20/20固定分组；从不随机拆分同一病人的patch。"""
    result = metadata.copy()
    result["partition"] = ""
    for center, frame in result.groupby("center", sort=True):
        patients = np.sort(frame["patient"].unique())
        if len(patients) < 3:
            raise ValueError(f"医院 {center} 只有 {len(patients)} 位病人，无法三路隔离。")
        patients = np.random.default_rng(seed + int(center)).permutation(patients)
        n_train = min(len(patients)-2, max(1, round(len(patients)*0.6)))
        n_val = min(len(patients)-n_train-1, max(1, round(len(patients)*0.2)))
        boundaries = (0, n_train, n_train+n_val, len(patients))
        for name, start, end in zip(["train", "val", "test"], boundaries[:-1], boundaries[1:]):
            mask = (result["center"] == center) & result["patient"].isin(patients[start:end])
            result.loc[mask, "partition"] = name
    return result


def sample_metadata(metadata, per_slide_class=128, seed=42):
    """None保留每一张图片；正整数仍为兼容的每切片每类上限。"""
    if per_slide_class is None:
        return metadata.sort_values(["center", "patient", "slide", "label", "image_id"]).reset_index(drop=True).copy()
    if per_slide_class < 1:
        raise ValueError("per_slide_class 必须大于0。")
    selected = []
    for key, group in metadata.groupby(["center", "patient", "slide", "label"], sort=True):
        key = tuple(int(value) for value in key)
        digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
        random = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        indices = random.choice(len(group), size=min(per_slide_class, len(group)), replace=False)
        selected.append(group.iloc[np.sort(indices)])
    return pd.concat(selected).sort_values(["center", "patient", "slide", "label", "image_id"]).reset_index(drop=True)


def _download_shard(root, revision, shard):
    _validate_revision(revision)
    target = _local_path(root, Path("data") / "raw" / revision / Path(shard).name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # 检查页脚可解析，避免把上次未完成的下载当成有效数据。
        pq.ParquetFile(target)
        return target
    temporary = target.with_suffix(".parquet.partial")
    for attempt in range(4):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else None
        try:
            response = _get(_url(revision, shard), headers=headers, stream=True)
            resumed = offset > 0 and response.status_code == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError("续传返回起点错误。")
            expected = int(response.headers.get("Content-Length", 0)) + (offset if resumed else 0)
            with response, temporary.open("ab" if resumed else "wb") as output:
                for block in response.iter_content(8 * 1024 * 1024):
                    output.write(block)
            if expected and temporary.stat().st_size != expected:
                raise RuntimeError(f"分片长度不完整: {shard}")
            break
        except (requests.RequestException, RuntimeError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    pq.ParquetFile(temporary)
    temporary.replace(target)
    print(f"image shard cached: {Path(shard).name}", flush=True)
    return target


def _save_image(image_bytes, target):
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.size != (96, 96):
            raise ValueError(f"原始patch大小不是96x96: {image.size}")
        temporary = target.with_suffix(".png.partial")
        image.convert("RGB").save(temporary, format="PNG")
        temporary.replace(target)


def _fetch_range_image(root, row, target, revision):
    """小检查直接取所需100行rowgroup，避免Viewer服务的频率限制。"""
    local = root/"data"/"raw"/revision/Path(row["_shard"]).name
    remote = None if local.exists() else _RangeFile(_url(revision, row["_shard"]))
    parquet = pq.ParquetFile(local if remote is None else remote)
    offset = 0
    position = int(row["_row_in_shard"])
    for i in range(parquet.num_row_groups):
        group = parquet.metadata.row_group(i)
        if offset <= position < offset+group.num_rows:
            if remote is not None:
                columns = [group.column(j) for j in range(group.num_columns)]
                starts = [min(c.data_page_offset, c.dictionary_page_offset or c.data_page_offset) for c in columns]
                start = min(starts)
                end = max(s+c.total_compressed_size for s, c in zip(starts, columns))
                remote.fetch(start, end-start)
            actual = parquet.read_row_group(i, columns=["image"]+META_COLUMNS, use_threads=False).slice(position-offset, 1).to_pylist()[0]
            for column in META_COLUMNS:
                if int(actual[column]) != int(row[column]):
                    raise ValueError(f"图像rowgroup与metadata不一致: {column}")
            _save_image(actual["image"]["bytes"], target)
            return
        offset += group.num_rows
    raise ValueError("分片内行索引越界。")


def _image_paths(selected, revision):
    """物理patch身份与划分seed无关；三套manifest共享同一份图像。"""
    _validate_revision(revision)
    selected = selected.copy()
    selected["sample_id"] = [f"c{int(r.center)}_p{int(r.patient)}_n{int(r.node)}_x{int(r.x_coord)}_y{int(r.y_coord)}"
                             for r in selected.itertuples()]
    selected["path"] = selected["sample_id"].map(lambda x: f"data/images/{revision}/{x}.png")
    if selected.sample_id.duplicated().any() or selected.path.duplicated().any():
        raise ValueError("重复物理patch，不能构造唯一图像缓存。")
    return selected


def _cache_images(root, selected, revision, fixture=False):
    _validate_revision(revision)
    root = Path(root).resolve()
    images = _local_path(root, Path("data") / "images" / revision)
    images.mkdir(parents=True, exist_ok=True)
    selected = _image_paths(selected, revision)
    missing = selected.loc[~selected["path"].map(lambda p: (root/p).exists())]
    if fixture:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_fetch_range_image, root, row, root/row["path"], revision)
                       for _, row in missing.iterrows()]
            for future in as_completed(futures):
                future.result()
    elif len(missing):
        shards = sorted(missing["_shard"].unique())
        with ThreadPoolExecutor(max_workers=6) as pool:
            paths = dict(zip(shards, pool.map(lambda s: _download_shard(root, revision, s), shards)))
        for shard, group in missing.groupby("_shard"):
            wanted = {int(r["_row_in_shard"]): root/r["path"] for _, r in group.iterrows()}
            positions = np.asarray(sorted(wanted), dtype=np.int64)
            parquet = pq.ParquetFile(paths[shard])
            offset = 0
            for i in range(parquet.num_row_groups):
                count = parquet.metadata.row_group(i).num_rows
                begin, end = np.searchsorted(positions, [offset, offset+count])
                indices = positions[begin:end]
                if len(indices):
                    values = parquet.read_row_group(i, columns=["image"]).column("image")
                    for idx in indices:
                        _save_image(values[idx-offset].as_py()["bytes"], wanted[idx])
                offset += count
    if not selected["path"].map(lambda p: (root/p).is_file()).all():
        raise RuntimeError("缓存仍缺少所选图片，不能把manifest标为图像准备完成。")
    return selected


def _group_summary(metadata):
    return [{"center": int(center), "patches": len(group), "patients": group.patient.nunique(),
             "slides": group.slide.nunique(), "class_counts": {str(k): int(v) for k, v in group.label.value_counts().items()}}
            for center, group in metadata.groupby("center")]


def prepare_data(root, per_slide_class=128, seed=42, revision=None, *, fixture_per_center=None,
                 partition_seed=None, manifest_path=None, metadata_only=False,
                 expected_images=EXPECTED_FULL_IMAGES):
    """准备真实缓存与manifest；旧调用保持兼容。

    per_slide_class=None表示全部导出图片参与互斥的train/val/test，不是全部用来训。
    partition_seed独立于抽样seed；manifest_path可放data/full/split_42/manifest.csv。
    metadata_only只写元数据，不创建/下载图像。三个划分复用按revision+patch身份
    命名的图像缓存；一次性图片hash清单由训练模块负责，不在此重复读取所有图片。
    expected_images默认锁定455954；小型离线测试可显式传其夹具规模。
    """
    root = Path(root).resolve()
    revision = revision or PINNED_REVISION
    _validate_revision(revision)
    partition_seed = int(seed if partition_seed is None else partition_seed)
    filename = "manifest_fixture.csv" if fixture_per_center is not None else "manifest.csv"
    manifest_path = _local_path(root, manifest_path if manifest_path is not None else Path("data")/filename)
    if manifest_path.suffix != ".csv":
        raise ValueError("manifest_path必须是项目内部的.csv文件。")
    output = manifest_path.parent
    metadata = scan_metadata(root, revision)
    if per_slide_class is None:
        _validate_full_export(metadata, expected_images)
    partitioned = assign_partitions(metadata, partition_seed)
    review = partition_review(partitioned, partition_seed)
    output.mkdir(parents=True, exist_ok=True)
    (output/"partition_review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    if not review["ready"]:
        raise ValueError("预先指定的患者划分缺类或发生泄漏，见partition_review.json；不会自动重挑seed。")
    selected = sample_metadata(partitioned, per_slide_class, seed)
    # 仅用于本机端到端检查，仍取真实patch；完整Colab实验不设置该参数。
    if fixture_per_center is not None:
        bounded = []
        for _, hospital in selected.groupby("center"):
            # 每个partition的每类至少两张，保证小检查能算AUC并运行分层子采样。
            coverage = pd.concat([g.sample(min(len(g), 2), random_state=seed)
                                  for _, g in hospital.groupby(["partition", "label"])])
            if fixture_per_center < len(coverage):
                raise ValueError(f"fixture_per_center 至少 {len(coverage)}，才能覆盖全部partition和类别。")
            rest = hospital.drop(coverage.index)
            extra = rest.sample(min(len(rest), fixture_per_center-len(coverage)), random_state=seed)
            bounded.append(pd.concat([coverage, extra]))
        selected = pd.concat(bounded).reset_index(drop=True)
    selected = (_image_paths(selected, revision) if metadata_only else
                _cache_images(root, selected, revision, fixture=fixture_per_center is not None))
    selected[MANIFEST_COLUMNS].to_csv(manifest_path, index=False)
    selected.to_parquet(output/("selected_fixture.parquet" if fixture_per_center else "selected.parquet"), index=False)
    metadata.groupby(["center", "patient", "slide", "label"]).size().rename("patches").to_csv(output/"metadata_group_counts.csv")
    group_map = partitioned[["center", "patient", "partition"]].drop_duplicates().sort_values(["center", "patient"])
    group_map.to_csv(output/"patient_partitions.csv", index=False)
    provenance = {
        "repository": REPOSITORY, "revision": revision, "protocol": "custom patient-disjoint per-hospital 60/20/20; not official WILDS",
        "seed": seed, "partition_seed": partition_seed, "per_slide_class_cap": per_slide_class, "fixture_per_center": fixture_per_center,
        "sampling": ("All exported images retained with natural label proportions; each image belongs to exactly one train/val/test partition."
                     if per_slide_class is None else "Uniform without replacement within each hospital/patient/slide/class; cap applied before fixture subsampling. Natural label proportions are changed by the per-class cap."),
        "selection_mode": "all_exported_images" if per_slide_class is None and fixture_per_center is None else "sampled_or_fixture",
        "images_cached": not metadata_only, "metadata_only": metadata_only,
        "image_cache_shared_across_partition_seeds": True,
        "source_policy": "source.train for fitting; source.val for checkpoint/hyperparameter selection; source.test for ID evaluation",
        "target_policy": ("target.train features only for distance analysis; target.val unused; target.test strictly held out"
                          if per_slide_class is None else "target.train images only for unlabeled adaptation; target.val unused; target.test strictly held out"),
        "official_export_counts": {str(k): int(v) for k, v in metadata._official_split.value_counts().items()},
        "official_export_center_counts": {str(split): {str(c): int(n) for c, n in g.center.value_counts().items()}
                                          for split, g in metadata.groupby("_official_split")},
        "warning_hf_validation": "HF validation merges source ID validation patches with center-1 OOD patches; never infer hospital from split name.",
        "all_metadata_summary": _group_summary(metadata), "selected_summary": _group_summary(selected),
        "manifest": str(manifest_path.relative_to(root)), "selected_images": len(selected),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256((root/"data"/f"metadata-{revision}.parquet").read_bytes()).hexdigest(),
        "readme": f"https://huggingface.co/datasets/{REPOSITORY}/blob/{revision}/README.md",
        "acquisition": "Metadata-only HTTP Range projection; full immutable parquet cache for Colab image extraction. Fixture uses revision-validated HF viewer cache images or immutable rowgroup Range patches; both checked against all 8 metadata fields.",
        "patient_partition_counts": {str(c): g.groupby("partition").patient.nunique().to_dict() for c, g in group_map.groupby("center")},
    }
    name = "provenance_fixture.json" if fixture_per_center is not None else "provenance.json"
    (output/name).write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance["all_metadata_summary"], indent=2), flush=True)
    print(f"Saved {manifest_path}: {len(selected)} real images", flush=True)
    return manifest_path


def prepare_full_manifests(root, partition_seeds=(42, 43, 44), revision=None,
                           output_dir="data/full", *, expected_images=EXPECTED_FULL_IMAGES):
    """只读全量metadata，生成预先指定划分；绝不调用图像获取代码。

    返回manifest_paths(str seed -> absolute CSV path)、reviews、all_ready与总报告路径。
    每份manifest恰含全部455954图片，只有partition列随seed变化；PNG路径相同。
    某seed缺类仍保留其原始划分并报告all_ready=False，绝不搜索替代seed。
    """
    root = Path(root).resolve()
    revision = revision or PINNED_REVISION
    _validate_revision(revision)
    output = _local_path(root, output_dir)
    seeds = [int(seed) for seed in partition_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("partition_seeds必须非空且不重复。")
    metadata = scan_metadata(root, revision)
    _validate_full_export(metadata, expected_images)
    base = _image_paths(sample_metadata(metadata, None), revision)
    output.mkdir(parents=True, exist_ok=True)
    reports, paths = [], {}
    for seed in seeds:
        partitioned = assign_partitions(base, seed)
        review = partition_review(partitioned, seed)
        folder = _local_path(root, output/str(f"split_{seed}"))
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/"manifest.csv"
        partitioned[MANIFEST_COLUMNS].to_csv(path, index=False)
        partitioned[["center", "patient", "partition"]].drop_duplicates().sort_values(["center", "patient"]).to_csv(folder/"patient_partitions.csv", index=False)
        review.update({"manifest_path": str(path), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "images_cached_by_this_operation": False})
        (folder/"partition_review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
        provenance = {"repository": REPOSITORY, "revision": revision, "selected_images": len(partitioned),
            "per_slide_class_cap": None, "partition_seed": seed, "selection_mode": "all_exported_images",
            "manifest": str(path.relative_to(root)), "manifest_sha256": review["manifest_sha256"],
            "protocol": "custom patient-disjoint per-hospital 60/20/20; not official WILDS",
            "source_policy": "source.train for fitting; source.val for checkpoint/hyperparameter selection; source.test for ID evaluation",
            "target_policy": "target.train features only for distance analysis; target.val unused; target.test strictly held out",
            "images_cached_by_this_operation": False, "metadata_only": True,
            "image_cache_shared_across_partition_seeds": True,
            "cache_validation": "metadata-only generation; every run must validate all shared images against the training module digest inventory",
            "selected_summary": _group_summary(partitioned)}
        (folder/"provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        reports.append(review)
        paths[str(seed)] = str(path)
    result = {"repository": REPOSITORY, "revision": revision, "total_images": len(base),
              "per_slide_class": None, "partition_seeds": seeds, "manifest_paths": paths,
              "all_ready": all(review["ready"] for review in reports), "reviews": reports,
              "image_cache": f"data/images/{revision}", "images_downloaded": False,
              "cache_policy": "same image IDs/paths for every partition seed; prepare images once and reuse; image digest belongs to training module"}
    report_path = output/"full_split_review.json"
    result["review_path"] = str(report_path)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_manifest(root, filename="manifest.csv"):
    path = Path(root)/"data"/filename
    frame = pd.read_csv(path)
    if not set(MANIFEST_COLUMNS).issubset(frame.columns):
        raise ValueError("Manifest缺少必要字段。")
    if frame.sample_id.duplicated().any() or frame.path.duplicated().any():
        raise ValueError("Manifest含重复patch。")
    if frame[MANIFEST_COLUMNS].isna().any().any() or not set(frame.label.unique()).issubset({0, 1}):
        raise ValueError("Manifest元数据缺失或标签不是0/1。")
    if (frame.groupby(["center", "patient"]).partition.nunique() > 1).any():
        raise ValueError("同一病人出现在多个partition。")
    if (frame.groupby(["center", "slide"]).partition.nunique() > 1).any():
        raise ValueError("同一切片出现在多个partition。")
    if not set(frame.partition).issubset({"train", "val", "test"}):
        raise ValueError("非法partition。")
    root = Path(root).resolve()
    for relative in frame.path:
        absolute = (root/relative).resolve()
        if Path(relative).is_absolute() or not absolute.is_relative_to(root) or not absolute.is_file():
            raise ValueError(f"图像缺失或路径越界: {relative}")
    return frame


def make_transform(image_size=96, augment=False):
    operations = [transforms.Resize((image_size, image_size))]
    if augment:
        operations += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation((0, 360))]
    operations += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(operations)


class PatchDataset(Dataset):
    def __init__(self, root, frame, image_size=96, augment=False):
        self.root = Path(root)
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = make_transform(image_size, augment)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.root/row.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row.label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--per-slide-class", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--fixture-per-center", type=int)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.metadata_only:
        metadata = scan_metadata(args.root, args.revision)
        print(json.dumps(_group_summary(metadata), indent=2))
        assign_partitions(metadata, args.seed)[["center", "patient", "partition"]].drop_duplicates().to_csv(Path(args.root)/"data"/"patient_partitions.csv", index=False)
        metadata.groupby(["center", "patient", "slide", "label"]).size().rename("patches").to_csv(Path(args.root)/"data"/"metadata_group_counts.csv")
    else:
        prepare_data(args.root, args.per_slide_class, args.seed, args.revision, fixture_per_center=args.fixture_per_center)


if __name__ == "__main__":
    main()
