"""Synthetic boundary tests: no downloads or course data required."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from camelyon.analysis import (
    compute_shift, coral_transform, extract_checkpoint_features, extract_reference_features,
    marginal_wasserstein, predict_adaptation_model, run_adaptation,
)


def feature_fixture(path, centers=2, dimension=6):
    rng = np.random.default_rng(8)
    rows, features = [], []
    for center in range(centers):
        for partition, count in (("train", 16), ("val", 8), ("test", 8)):
            for i in range(count):
                label = i % 2
                sample = f"c{center}_{partition}_{i}"
                rows.append(dict(sample_id=sample, center=str(center), patient=sample,
                                 slide=sample, label=label, partition=partition))
                value = rng.normal(0, .3, dimension)
                value[0] += 2 * label - 1
                value += .2 * center
                features.append(value)
    arrays = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    np.savez(path, features=np.array(features), **arrays)
    return np.array(features), arrays


class ShiftTests(unittest.TestCase):
    def test_known_1d_shift_and_permutation_identity(self):
        x = np.array([[0., 2.], [1., 3.], [2., 4.]])
        self.assertEqual(marginal_wasserstein(x, x[::-1]), 0.)
        self.assertAlmostEqual(marginal_wasserstein(x, x + [2., 4.]), 3.)
        self.assertAlmostEqual(marginal_wasserstein(np.zeros((2, 3)), np.ones((7, 3))), 1.)

    def test_identical_centers_classconditional_and_test_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, arrays = feature_fixture(root / "original.npz", centers=5)
            # Conditional distributions identical; raw prevalence changes across centers.
            train_c1 = np.flatnonzero((arrays["center"] == "1") & (arrays["partition"] == "train"))
            arrays["label"][train_c1] = [0] * 4 + [1] * 12
            labels = arrays["label"]
            features = np.repeat(labels[:, None], 3, axis=1).astype(float)
            # Test features are deliberately extreme and must not enter shift.
            features[arrays["partition"] == "test"] += 1e7
            np.savez(root / "features.npz", features=features, **arrays)
            csv = compute_shift(root / "features.npz", root / "result")
            frame = pd.read_csv(csv)
            self.assertEqual(len(frame), 20)
            self.assertTrue((frame["shift_classconditional"] == 0).all())
            self.assertTrue((frame["shift_balanced"] == 0).all())
            self.assertGreater(frame["shift_raw"].max(), 0)

    def test_patient_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            features, arrays = feature_fixture(path)
            arrays["patient"][16] = arrays["patient"][0]  # train vs val
            np.savez(path, features=features, **arrays)
            with self.assertRaisesRegex(ValueError, "patient crosses"):
                compute_shift(path, directory)

    def test_coral_same_distribution_is_identity(self):
        rng = np.random.default_rng(3)
        features = rng.normal(size=(25, 6))
        transform, source_mean, target_mean = coral_transform(features, features.copy())
        np.testing.assert_allclose(transform, np.eye(6), atol=1e-12)
        np.testing.assert_allclose((features - source_mean) @ transform + target_mean, features, atol=1e-12)


class AdaptationTests(unittest.TestCase):
    def test_no_target_test_fit_or_target_train_labels_and_serialized_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features, arrays = feature_fixture(root / "original.npz")
            config = dict(sources=[0], targets=[1], feature_source="synthetic-fixed-backbone",
                          pca_components=3, C_candidates=[.1, 1.], seed=17)
            first = run_adaptation(root / "original.npz", root / "a", config)
            changed = features.copy()
            test = (arrays["center"] == "1") & (arrays["partition"] == "test")
            target_train = (arrays["center"] == "1") & (arrays["partition"] == "train")
            changed[test] = 100000  # must affect predictions, never the fitted model
            mutated = {key: value.copy() for key, value in arrays.items()}
            mutated["label"][test | target_train] = 1 - mutated["label"][test | target_train]
            np.savez(root / "mutated.npz", features=changed, **mutated)
            second = run_adaptation(root / "mutated.npz", root / "b", config)
            results_a, results_b = pd.read_csv(first["results_path"]), pd.read_csv(second["results_path"])
            for method in results_a["method"]:
                row_a = results_a[results_a["method"] == method].iloc[0]
                row_b = results_b[results_b["method"] == method].iloc[0]
                a, b = joblib.load(row_a.model_path), joblib.load(row_b.model_path)
                np.testing.assert_allclose(a["pca"].components_, b["pca"].components_)
                np.testing.assert_allclose(a["scaler"].mean_, b["scaler"].mean_)
                np.testing.assert_allclose(a["classifier"].coef_, b["classifier"].coef_)
                np.testing.assert_allclose(a["classifier"].intercept_, b["classifier"].intercept_)
                self.assertEqual(row_a.selected_C, row_b.selected_C)
                if method == "coral_logistic":
                    np.testing.assert_allclose(a["coral_transform"], b["coral_transform"])
                predictions = pd.read_csv(first["predictions_path"])
                recorded = predictions[predictions["method"] == method]["probability"]
                np.testing.assert_allclose(predict_adaptation_model(row_a.model_path, features[test]), recorded)
            metadata = json.loads(Path(first["metadata_path"]).read_text())
            pair = metadata["pairs"][0]
            self.assertFalse(metadata["target_labels_used_for_fitting"])
            self.assertFalse(set(pair["evaluation_sample_ids"]) & set(pair["projection_fit_sample_ids"]))
            self.assertFalse(set(pair["evaluation_sample_ids"]) & set(pair["target_unlabeled_fit_sample_ids"]))
            self.assertTrue(all("_train_" in item for item in pair["target_unlabeled_fit_sample_ids"]))
            self.assertTrue(all("_val_" in item for item in pair["selection_sample_ids"]))

    def test_shape_and_unavailable_center_fail(self):
        with self.assertRaisesRegex(ValueError, "matching dimensions"):
            marginal_wasserstein(np.zeros((2, 3)), np.zeros((2, 4)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            feature_fixture(path)
            with self.assertRaisesRegex(ValueError, "absent"):
                run_adaptation(path, directory, {"sources": [9]})


class ReferenceCacheTests(unittest.TestCase):
    def test_checkpoint_extraction_strict_load_and_shift_reference_separation(self):
        import torch
        import camelyon.models  # load torch registration modules before sys.modules isolation

        class FakeNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(2.))
                self.fc = torch.nn.Linear(2048, 2)
            def forward(self, images):
                pooled = images.mean(dim=(1, 2, 3))[:, None].repeat(1, 2048) * self.scale
                return self.fc(pooled)

        class FakeDataset(torch.utils.data.Dataset):
            def __init__(self, root, frame, **kwargs):
                self.frame = frame
            def __len__(self):
                return len(self.frame)
            def __getitem__(self, index):
                return torch.ones((3, 4, 4)), 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            pd.DataFrame([dict(sample_id=f"s{i}", path=f"i{i}.png", center=i,
                patient=f"p{i}", slide=f"w{i}", label=i, partition="train") for i in range(2)]).to_csv(root / "data/manifest.csv", index=False)
            network = FakeNetwork()
            checkpoint = root / "best.pt"
            torch.save({"state_dict": network.state_dict(), "spec": {"depth": "head", "seed": 42, "source": "0"}}, checkpoint)
            fake_module = types.ModuleType("camelyon.data")
            fake_module.PatchDataset = FakeDataset
            with patch.dict(sys.modules, {"camelyon.data": fake_module}), patch("camelyon.models.build_model", return_value=FakeNetwork()) as constructor:
                output = extract_checkpoint_features(root, {"device": "cpu"}, checkpoint, root / "out", "trained_head")
                constructor.assert_called_once_with("head", seed=42, pretrained=False)
                with np.load(output["feature_path"]) as archive:
                    self.assertEqual(archive["features"].shape, (2, 2048))
                    np.testing.assert_allclose(archive["features"], 2.)
                with self.assertRaisesRegex(ValueError, "fixed ImageNet reference"):
                    compute_shift(output["feature_path"], root / "badshift")

    def test_fixed_shape_cache_identity_and_no_second_extraction(self):
        import torch
        import torchvision.models  # avoid reimporting Torch libraries after mock restoration

        class FakeResNet(torch.nn.Module):
            def forward(self, images):
                return images.mean(dim=(1, 2, 3))[:, None].repeat(1, 2048)

        class FakeDataset(torch.utils.data.Dataset):
            def __init__(self, root, frame, image_size=96, augment=False):
                assert augment is False
                self.frame = frame
            def __len__(self):
                return len(self.frame)
            def __getitem__(self, index):
                return torch.full((3, 4, 4), float(index)), 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            manifest = pd.DataFrame([dict(sample_id=f"sample{i}", path=f"image{i}.png", center=i,
                patient=f"p{i}", slide=f"s{i}", label=i, partition="train") for i in range(2)])
            manifest.to_csv(root / "data/manifest.csv", index=False)
            fake_module = types.ModuleType("camelyon.data")
            fake_module.PatchDataset = FakeDataset
            with patch.dict(sys.modules, {"camelyon.data": fake_module}), patch("torchvision.models.resnet50", return_value=FakeResNet()) as constructor:
                result = extract_reference_features(root, {"device": "cpu"}, root / "features")
                second = extract_reference_features(root, {"device": "cpu"}, root / "features")
                self.assertEqual(constructor.call_count, 1)
                self.assertFalse(result["reused"])
                self.assertTrue(second["reused"])
                with np.load(result["feature_path"]) as archive:
                    self.assertEqual(archive["features"].shape, (2, 2048))
                    np.testing.assert_allclose(archive["features"][:, 0], [0, 1])
                manifest.loc[0, "path"] = "changed.png"
                manifest.to_csv(root / "data/manifest.csv", index=False)
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    extract_reference_features(root, {"device": "cpu"}, root / "features")


if __name__ == "__main__":
    unittest.main()
