"""数据协议的实质性回归：病人隔离、确定性、抽样上限与真实图像接口。"""
import pandas as pd
import hashlib
import io
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from camelyon.data import (MANIFEST_COLUMNS, PatchDataset, assign_partitions,
                           load_manifest, sample_metadata, prepare_data,
                           prepare_full_manifests, partition_review, _image_paths)


def metadata():
    rows = []
    for center in range(5):
        for patient in range(5):
            for slide in range(2):
                for label in range(2):
                    for patch in range(7):
                        rows.append({"center": center, "patient": patient,
                                     "slide": patient*2+slide, "label": label,
                                     "image_id": len(rows), "node": slide,
                                     "x_coord": label*100+patch, "y_coord": 0})
    return pd.DataFrame(rows)


class DataTests(unittest.TestCase):
    def test_none_cap_preserves_every_image_and_natural_class_counts(self):
        original = assign_partitions(metadata()).query("label == 0 or image_id % 3 == 0")
        selected = sample_metadata(original, per_slide_class=None, seed=42)
        self.assertEqual(len(selected), len(original))
        self.assertEqual(set(selected.image_id), set(original.image_id))
        self.assertEqual(selected.label.value_counts().to_dict(), original.label.value_counts().to_dict())
        pd.testing.assert_frame_equal(selected, sample_metadata(original, per_slide_class=None, seed=44))

    def test_full_metadata_manifests_share_paths_without_downloading_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m = metadata()
            with patch("camelyon.data.scan_metadata", return_value=m), \
                 patch("camelyon.data._cache_images", side_effect=AssertionError("metadata-only must not acquire images")):
                result = prepare_full_manifests(root, expected_images=len(m))
            self.assertTrue(result["all_ready"])
            self.assertFalse(result["images_downloaded"])
            frames = [pd.read_csv(result["manifest_paths"][str(seed)]) for seed in [42, 43, 44]]
            for frame in frames:
                self.assertEqual(len(frame), len(m))
                self.assertFalse(frame.sample_id.duplicated().any())
                self.assertEqual(frame.groupby(["center", "patient"]).partition.nunique().max(), 1)
                self.assertEqual(frame.groupby(["center", "slide"]).partition.nunique().max(), 1)
                self.assertTrue(frame.groupby(["center", "partition"]).label.nunique().eq(2).all())
                pd.testing.assert_frame_equal(frame.drop(columns="partition"), frames[0].drop(columns="partition"))
            self.assertFalse(frames[0].partition.equals(frames[1].partition))
            self.assertFalse((root/"data/images").exists())
            self.assertEqual(Path(result["manifest_paths"]["42"]),root.resolve()/"data/full/split_42/manifest.csv")
            for seed in [42,43,44]:
                provenance=json.loads((Path(result["manifest_paths"][str(seed)]).parent/"provenance.json").read_text())
                self.assertEqual(provenance["selected_images"],len(m))
                self.assertIsNone(provenance["per_slide_class_cap"])
                self.assertEqual(provenance["partition_seed"],seed)
                self.assertFalse(provenance["images_cached_by_this_operation"])
                self.assertIn("only for distance analysis",provenance["target_policy"])

    def test_full_class_failure_is_reported_without_replacing_predeclared_split(self):
        with tempfile.TemporaryDirectory() as directory:
            m = metadata()
            m.loc[m.patient.ne(0), "label"] = 0
            with patch("camelyon.data.scan_metadata", return_value=m):
                result = prepare_full_manifests(directory, partition_seeds=[42,43,44], expected_images=len(m))
            self.assertFalse(result["all_ready"])
            self.assertTrue(all(review["missing_class_groups"] for review in result["reviews"]))
            self.assertEqual(result["partition_seeds"], [42,43,44])
            actual = pd.read_csv(result["manifest_paths"]["42"])
            expected = assign_partitions(m,42)
            self.assertEqual(actual.groupby(["center","patient"]).partition.first().to_dict(),
                             expected.groupby(["center","patient"]).partition.first().to_dict())

    def test_full_count_output_paths_and_revision_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("camelyon.data.scan_metadata", return_value=metadata()):
                with self.assertRaisesRegex(ValueError,"455954"):
                    prepare_full_manifests(directory)
            with patch("camelyon.data.scan_metadata", side_effect=AssertionError("unsafe path must fail before data access")):
                with self.assertRaisesRegex(ValueError,"路径越界"):
                    prepare_full_manifests(directory,output_dir="../outside")
                with self.assertRaisesRegex(ValueError,"commit SHA"):
                    prepare_full_manifests(directory,revision="../../outside")

    def test_full_cache_is_materialized_once_and_reused_by_second_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/"data"
            data.mkdir()
            revision = "a"*40
            m = metadata().query("x_coord % 100 == 0").copy().reset_index(drop=True)
            m["_official_split"] = "train"
            m["_shard"] = "data/train-fixture.parquet"
            m["_row_in_shard"] = np.arange(len(m))
            m.to_parquet(data/f"metadata-{revision}.parquet",index=False)
            buffer = io.BytesIO()
            Image.new("RGB",(96,96),color=(123,45,67)).save(buffer,format="PNG")
            shard = data/"fixture_images.parquet"
            pq.write_table(pa.Table.from_pylist([{"image":{"bytes":buffer.getvalue()}} for _ in range(len(m))]),shard,row_group_size=11)
            with patch("camelyon.data._download_shard",return_value=shard) as download, patch("builtins.print"):
                first = prepare_data(root,None,revision=revision,partition_seed=42,
                    manifest_path="data/full/split_42/manifest.csv",expected_images=len(m))
                self.assertEqual(download.call_count,1)
                cached = list((data/"images"/revision).glob("*.png"))
                self.assertEqual(len(cached),len(m))
                before={path.name:(path.stat().st_mtime_ns,hashlib.sha256(path.read_bytes()).hexdigest()) for path in cached}
                second = prepare_data(root,None,revision=revision,partition_seed=43,
                    manifest_path="data/full/split_43/manifest.csv",expected_images=len(m))
                self.assertEqual(download.call_count,1)
            self.assertEqual(before,{path.name:(path.stat().st_mtime_ns,hashlib.sha256(path.read_bytes()).hexdigest()) for path in cached})
            left,right=pd.read_csv(first),pd.read_csv(second)
            pd.testing.assert_frame_equal(left.drop(columns="partition"),right.drop(columns="partition"))
            self.assertEqual(len(load_manifest(root,"full/split_43/manifest.csv")),len(m))
            provenance=json.loads((second.parent/"provenance.json").read_text())
            self.assertEqual(provenance["selection_mode"],"all_exported_images")
            self.assertIsNone(provenance["per_slide_class_cap"])
            self.assertEqual(provenance["partition_seed"],43)

    def test_patient_isolation_and_reproducibility(self):
        first = assign_partitions(metadata(), seed=42)
        second = assign_partitions(metadata().sample(frac=1, random_state=1), seed=42)
        assert first.groupby(["center", "patient"]).partition.nunique().max() == 1
        for _, group in first.groupby("center"):
            assert group.groupby("partition").patient.nunique().to_dict() == {"test": 1, "train": 3, "val": 1}
        key = ["center", "patient"]
        assert first.groupby(key).partition.first().equals(second.groupby(key).partition.first())


    def test_cap_applies_per_slide_class_and_preserves_all_groups(self):
        data = assign_partitions(metadata())
        first = sample_metadata(data, per_slide_class=3)
        second = sample_metadata(data, per_slide_class=3)
        pd.testing.assert_frame_equal(first, second)
        assert first.groupby(["center", "patient", "slide", "label"]).size().eq(3).all()
        assert len(first.groupby(["center", "patient", "slide", "label"])) == len(data.groupby(["center", "patient", "slide", "label"]))
        assert not first.image_id.duplicated().any()


    def test_too_few_patients_fails_instead_of_patch_splitting(self):
        data = metadata().query("patient < 2")
        with self.assertRaisesRegex(ValueError, "无法三路隔离"):
            assign_partitions(data)


    def test_manifest_image_interface_and_leak_detection(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        directory = tmp_path/"data"
        directory.mkdir()
        Image.new("RGB", (96, 96), color=(120, 60, 80)).save(directory/"real_interface.png")
        frame = pd.DataFrame([["sample", "data/real_interface.png", 0, 4, 0, 1, "train"]], columns=MANIFEST_COLUMNS)
        frame.to_csv(directory/"manifest.csv", index=False)
        loaded = load_manifest(tmp_path)
        tensor, label = PatchDataset(tmp_path, loaded)[0]
        assert tuple(tensor.shape) == (3, 96, 96)
        assert label == 1
        leaking = pd.concat([frame, frame.assign(sample_id="second", path="data/second.png", partition="test")])
        leaking.to_csv(directory/"manifest.csv", index=False)
        with self.assertRaisesRegex(ValueError, "同一病人"):
            load_manifest(tmp_path)


    def test_manifest_rejects_path_outside_root(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        directory = tmp_path/"data"
        directory.mkdir()
        frame = pd.DataFrame([["sample", "../outside.png", 0, 4, 0, 1, "test"]], columns=MANIFEST_COLUMNS)
        frame.to_csv(directory/"manifest.csv", index=False)
        with self.assertRaisesRegex(ValueError, "路径越界"):
            load_manifest(tmp_path)


if __name__ == "__main__":
    unittest.main()
