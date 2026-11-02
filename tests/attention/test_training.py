import json
import tempfile
import unittest
from pathlib import Path

import torch

from ml.models.attention.components import CustomTransformerClassifier
from ml.models.attention.inference import predict
from ml.models.attention.report import _verify_and_recompute
from ml.models.attention.train import atomic_torch, load_model_checkpoint, run_epoch


class AttentionTrainingTests(unittest.TestCase):
    def test_tiny_batch_overfit_and_cpu_training(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(19)
            model = CustomTransformerClassifier(
                vocab_size=24,
                model_dim=16,
                num_heads=4,
                num_layers=1,
                feedforward_dim=32,
                num_classes=4,
                dropout=0.0,
                max_length=6,
            )
            sequences = torch.tensor(
                [
                    [1, 2, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0],
                    [3, 4, 0, 0, 0, 0], [4, 3, 0, 0, 0, 0],
                    [5, 6, 0, 0, 0, 0], [6, 5, 0, 0, 0, 0],
                    [7, 8, 0, 0, 0, 0], [8, 7, 0, 0, 0, 0],
                ],
                dtype=torch.long,
            )
            labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            first = run_epoch(model, sequences, labels, torch.device("cpu"), 8)
            for epoch in range(24):
                result = run_epoch(
                    model,
                    sequences,
                    labels,
                    torch.device("cpu"),
                    8,
                    optimizer,
                    seed=epoch + 4,
                )
            self.assertLess(result["loss"], first["loss"])
            self.assertGreaterEqual(result["macro_f1"], 0.99)
        finally:
            torch.set_num_threads(previous_threads)

    def test_checkpoint_reload_and_phase2_preprocessing_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "artifacts"
            for folder in (data_root / "phase1", data_root / "phase2"):
                folder.mkdir(parents=True)
            vocabulary = {"<PAD>": 0, "<UNK>": 1, "paper": 2, "vision": 3}
            (data_root / "phase2/vocabulary.json").write_text(json.dumps(vocabulary), encoding="utf-8")
            (data_root / "phase2/metadata.json").write_text(
                json.dumps({"sequence_length": 6}), encoding="utf-8"
            )
            labels = {"0": "Algorithms", "1": "NLP/AI", "2": "Systems/Infra", "3": "Vision"}
            (data_root / "phase1/metadata.json").write_text(
                json.dumps({"label_mapping": labels}), encoding="utf-8"
            )

            torch.manual_seed(11)
            config = {
                "vocab_size": len(vocabulary),
                "model_dim": 8,
                "num_heads": 2,
                "num_layers": 1,
                "feedforward_dim": 16,
                "num_classes": 4,
                "dropout": 0.0,
                "max_length": 6,
                "pad_idx": 0,
            }
            original = CustomTransformerClassifier(**config).eval()
            checkpoint_path = root / "best.pt"
            atomic_torch(
                checkpoint_path,
                {"model_config": config, "model_state_dict": original.state_dict(), "epoch": 1},
            )
            reloaded, _ = load_model_checkpoint(checkpoint_path, "cpu")
            sample = torch.tensor([[2, 3, 0, 0, 0, 0]], dtype=torch.long)
            with torch.no_grad():
                self.assertTrue(torch.equal(original(sample), reloaded(sample)))

            result = predict(
                "Paper",
                "Vision paper",
                checkpoint_path=checkpoint_path,
                root=data_root,
                device="cpu",
            )
            self.assertIn(result["category"], labels.values())
            self.assertEqual(set(result["probabilities"]), set(labels.values()))
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=6)
            self.assertEqual(result["device"], "cpu")

    def test_invalid_epoch_input(self):
        model = CustomTransformerClassifier(8, model_dim=8, num_heads=2, max_length=4)
        with self.assertRaises(ValueError):
            run_epoch(
                model,
                torch.empty((0, 4), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
                torch.device("cpu"),
                2,
            )

    def test_report_recomputes_saved_metrics_and_checks_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.json"
            predictions_path = root / "predictions.json"
            predictions = [
                {"arxiv_id": str(index), "actual": index, "predicted": index}
                for index in range(4)
            ]
            report = {
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "weighted_f1": 1.0,
            }
            metrics_path.write_text(json.dumps(report), encoding="utf-8")
            predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
            observed, mapped = _verify_and_recompute(
                metrics_path, predictions_path, {"0", "1", "2", "3"}
            )
            self.assertEqual(len(mapped), 4)
            self.assertEqual(observed["recomputed_report"]["macro avg"]["f1-score"], 1.0)
            with self.assertRaises(ValueError):
                _verify_and_recompute(metrics_path, predictions_path, {"0", "1", "2"})


if __name__ == "__main__":
    unittest.main()
