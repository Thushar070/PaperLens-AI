from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml.models.pretrained.runtime import expose_existing_base_site_packages

expose_existing_base_site_packages()

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from ml.models.pretrained.classifier import ClassifierPredictor, load_classifier
from ml.models.pretrained.data import FIELDS, collate_examples, paper_text, tokenize_records
from ml.models.pretrained.embeddings import encode_query, generate_embeddings
from ml.models.pretrained.train import run_experiment
from ml.retrieval.transformer import TransformerRetriever


def make_record(identifier: str, label: int, title: str = "hello graph", abstract: str = "world research paper") -> dict:
    return {"arxiv_id": identifier, "title": title, "abstract": abstract, "primary_cat": "cs.AI", "all_cats": "cs.AI | cs.LG", "label": label}


class FakeSentenceEncoder:
    max_seq_length = 16

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        return np.asarray([[1.0, 0.0] if "graph" in text.lower() else [0.0, 1.0] for text in texts], dtype=np.float32)


class Phase7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_root.name)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "graph", "world", "research", "paper", "vision", "systems", "nlp", "."]
        cls.model_dir = cls.root / "tiny_pretrained"
        cls.model_dir.mkdir()
        vocab_path = cls.model_dir / "vocab.txt"
        vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
        tokenizer.save_pretrained(cls.model_dir)
        config = BertConfig(vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1, num_attention_heads=4, intermediate_size=48, max_position_embeddings=32, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        BertModel(config).save_pretrained(cls.model_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temp_root.cleanup()

    def test_tokenization_handles_empty_long_batch_and_masks(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        rows = [make_record("one", 0), make_record("two", 1, title="", abstract=None), make_record("three", 2, title="hello " * 30, abstract="world " * 30)]
        tokenized = tokenize_records(tokenizer, rows, max_length=8)
        self.assertEqual(len(tokenized), 3)
        self.assertEqual(len(tokenized[2]["input_ids"]), 8)
        self.assertEqual(tokenized[1]["input_ids"], [tokenizer.cls_token_id, tokenizer.sep_token_id])
        self.assertTrue(all(len(item["input_ids"]) == len(item["attention_mask"]) for item in tokenized))
        batch = collate_examples(tokenized[:2], tokenizer.pad_token_id, torch)
        self.assertEqual(tuple(batch["input_ids"].shape), tuple(batch["attention_mask"].shape))
        self.assertEqual(batch["attention_mask"][1].sum().item(), 2)
        self.assertEqual(set(batch["arxiv_ids"]), {"one", "two"})

    def test_paper_text_and_validation_preserve_source_values(self):
        row = make_record("raw", 0, title=" title ", abstract=" abstract ")
        self.assertEqual(paper_text(row), "title\n\nabstract")
        self.assertEqual(tuple(row), FIELDS)
        with self.assertRaises(ValueError):
            tokenize_records(object(), [make_record("bad", 5)], 8)
        with self.assertRaises(TypeError):
            tokenize_records(object(), ["malformed"], 8)

    def test_classifier_has_four_outputs_and_label_mapping(self):
        model = load_classifier(str(self.model_dir))
        self.assertEqual(model.config.num_labels, 4)
        self.assertEqual(model.config.id2label, {0: "Algorithms", 1: "NLP/AI", 2: "Systems/Infra", 3: "Vision"})
        model.eval()
        batch = torch.zeros((2, 4), dtype=torch.long)
        with torch.no_grad():
            output = model(input_ids=batch, attention_mask=torch.ones_like(batch)).logits
        self.assertEqual(tuple(output.shape), (2, 4))

    def test_training_smoke_checkpoint_round_trip_cpu(self):
        import ml.models.pretrained.train as training

        train = [make_record(f"tr-{index}", index % 4) for index in range(8)]
        validation = [make_record(f"va-{index}", index) for index in range(4)]
        with tempfile.TemporaryDirectory() as temporary, patch.object(training, "update_checkpoint", lambda *args, **kwargs: {}):
            report = run_experiment(
                model_path=self.model_dir,
                output_dir=Path(temporary) / "smoke",
                train_rows=train,
                validation_rows=validation,
                config={"name": "smoke", "learning_rate": 0.001, "batch_size": 4, "gradient_accumulation_steps": 1, "epochs": 1, "early_stopping_patience": 1, "weight_decay": 0.0, "max_length": 16, "mixed_precision": False, "seed": 7},
                device_override="cpu",
            )
            self.assertEqual(report["status"], "complete")
            predictor = ClassifierPredictor(report["best_checkpoint"], device="cpu")
            one = predictor.predict_one(validation[0])
            multiple = predictor.predict(validation, batch_size=2)
            self.assertEqual(one["arxiv_id"], validation[0]["arxiv_id"])
            self.assertIn(one["predicted_label_name"], {"Algorithms", "NLP/AI", "Systems/Infra", "Vision"})
            self.assertEqual(len(one["probabilities"]), 4)
            self.assertTrue(np.allclose(one["probabilities"], multiple[0]["probabilities"], atol=1e-6))
            self.assertEqual(len(multiple), len(validation))
            self.assertTrue(all(tuple(row[field] for field in FIELDS) == tuple(source[field] for field in FIELDS) for row, source in zip(multiple, validation)))
            reloaded = ClassifierPredictor(report["best_checkpoint"], device="cpu")
            self.assertEqual(reloaded.predict(validation, batch_size=2), multiple)
            if torch.cuda.is_available():
                cuda_predictor = ClassifierPredictor(report["best_checkpoint"], device="cuda:0")
                cuda_prediction = cuda_predictor.predict_one(validation[0])
                self.assertEqual(cuda_prediction["predicted_label"], one["predicted_label"])
                self.assertTrue(np.allclose(cuda_prediction["probabilities"], one["probabilities"], atol=1e-5))

    def test_incremental_embedding_generation_alignment_and_integrity(self):
        import ml.models.pretrained.embeddings as embedding_module

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            rows = {
                "train": [make_record("a", 0), make_record("b", 1, title="vision")],
                "validation": [make_record("c", 2, title="systems")],
                "test": [make_record("d", 3, title="nlp")],
            }
            (source / "metadata.json").write_text(json.dumps({"split_sizes": {key: len(value) for key, value in rows.items()}}), encoding="utf-8")
            for split, values in rows.items():
                (source / f"{split}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in values), encoding="utf-8")
            fake = FakeSentenceEncoder()
            with patch.object(embedding_module, "load_sentence_encoder", lambda *args, **kwargs: (fake, "cpu")), patch.object(embedding_module, "update_checkpoint", lambda *args, **kwargs: {}):
                output = root / "output"
                result = generate_embeddings(source, output, model_path=root, batch_size=2, flush_every_batches=1)
            matrix = np.load(output / "paper_embeddings.npy")
            index = [json.loads(line) for line in (output / "embedding_index.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["paper_count"], 4)
            self.assertEqual(matrix.shape, (4, 2))
            self.assertTrue(np.isfinite(matrix).all())
            self.assertEqual([row["arxiv_id"] for row in index], ["a", "b", "c", "d"])
            self.assertTrue(all(all(field in row for field in FIELDS) for row in index))
            self.assertEqual([row["embedding_offset"] for row in index], list(range(4)))
            self.assertGreater(result["embedding_norm_min"], 0)
            with patch.object(embedding_module, "load_sentence_encoder", lambda *args, **kwargs: (fake, "cpu")):
                retriever = TransformerRetriever.from_artifacts(output, model_path=root, device="cpu")
            stored_result = retriever.search("graph query", k=1)
            self.assertEqual(stored_result[0]["arxiv_id"], "a")
            self.assertEqual(stored_result[0]["label"], 0)

    def test_query_embedding_empty_malformed_and_finite(self):
        encoder = FakeSentenceEncoder()
        self.assertEqual(encode_query(encoder, "").size, 0)
        self.assertTrue(np.allclose(encode_query(encoder, "graph query"), [1.0, 0.0]))
        with self.assertRaises(TypeError):
            encode_query(encoder, None)

    def test_transformer_retrieval_cosine_invalid_ties_and_metadata(self):
        records = [make_record("b", 1), make_record("a", 0, title="vision"), make_record("invalid", 2)]
        vectors = np.asarray([[1, 0], [1, 0], [np.nan, 0]], dtype=np.float32)
        retriever = TransformerRetriever(records, vectors, FakeSentenceEncoder())
        result = retriever.search("graph query", k=5)
        self.assertEqual([row["arxiv_id"] for row in result], ["a", "b"])
        self.assertTrue(all(np.isclose(row["score"], 1.0) for row in result))
        self.assertTrue(all(all(field in row for field in FIELDS) for row in result))
        self.assertEqual(retriever.search("  ", 5), [])
        self.assertEqual(retriever.search("graph", 0), [])
        with self.assertRaises(TypeError):
            retriever.search(None, 1)
        with self.assertRaises(ValueError):
            retriever.search("graph", -1)
        with self.assertRaises(ValueError):
            TransformerRetriever(records, np.zeros((2, 2)), FakeSentenceEncoder())

    def test_transformer_retrieval_rejects_bad_query_embedding(self):
        class BadEncoder(FakeSentenceEncoder):
            def encode(self, texts, **kwargs):
                return np.asarray([[np.nan, 1]], dtype=np.float32)

        retriever = TransformerRetriever([make_record("one", 0)], np.asarray([[1, 0]], dtype=np.float32), BadEncoder())
        with self.assertRaises(ValueError):
            retriever.search("query", 1)


if __name__ == "__main__":
    unittest.main()
