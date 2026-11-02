import math
import unittest

import torch

from ml.models.attention.components import (
    CustomTransformerClassifier,
    MultiHeadSelfAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    ScaledDotProductAttention,
    TransformerEncoderBlock,
)


class AttentionComponentTests(unittest.TestCase):
    def test_sinusoidal_positions_and_fixed_buffer(self):
        encoding = PositionalEncoding(model_dim=4, max_length=5)
        values = encoding.encoding[0]
        self.assertEqual(tuple(values.shape), (5, 4))
        self.assertTrue(torch.equal(values[0], torch.tensor([0.0, 1.0, 0.0, 1.0])))
        self.assertAlmostEqual(float(values[1, 0]), math.sin(1.0), places=6)
        self.assertAlmostEqual(float(values[1, 1]), math.cos(1.0), places=6)
        self.assertAlmostEqual(float(values[1, 2]), math.sin(0.01), places=6)
        self.assertAlmostEqual(float(values[1, 3]), math.cos(0.01), places=6)
        self.assertFalse(encoding.encoding.requires_grad)
        self.assertFalse(dict(encoding.named_parameters()))
        self.assertEqual(tuple(encoding(torch.zeros(2, 3, 4)).shape), (2, 3, 4))
        with self.assertRaises(ValueError):
            encoding(torch.zeros(1, 6, 4))

    def test_scaled_attention_matches_hand_computed_reference(self):
        attention = ScaledDotProductAttention()
        query = torch.tensor([[[[1.0, 0.0]]]])
        key = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        value = torch.tensor([[[[2.0, 0.0], [0.0, 4.0]]]])
        output, weights = attention(query, key, value)
        logits = torch.tensor([0.0, 1.0 / math.sqrt(2.0)])
        expected_weights = torch.softmax(logits, dim=0)
        expected_output = expected_weights[0] * value[0, 0, 0] + expected_weights[1] * value[0, 0, 1]
        self.assertTrue(torch.allclose(weights[0, 0, 0], expected_weights, atol=1e-6))
        self.assertTrue(torch.allclose(output[0, 0, 0], expected_output, atol=1e-6))
        self.assertAlmostEqual(float(weights.sum(dim=-1).item()), 1.0, places=6)

    def test_key_and_query_padding_masks_and_empty_keys(self):
        attention = ScaledDotProductAttention()
        query = torch.zeros(2, 1, 2, 2)
        key = torch.zeros(2, 1, 3, 2)
        value = torch.tensor(
            [[[[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]]], [[[5.0, 7.0]] * 3]]
        )
        key_mask = torch.tensor([[True, True, False], [False, False, False]])
        query_mask = torch.tensor([[True, False], [True, True]])
        output, weights = attention(
            query, key, value, key_padding_mask=key_mask, query_padding_mask=query_mask
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.equal(weights[0, 0, :, 2], torch.zeros(2)))
        self.assertTrue(torch.allclose(output[0, 0, 0], torch.tensor([2.0, 0.0])))
        self.assertTrue(torch.equal(weights[0, 0, 1], torch.zeros(3)))
        self.assertTrue(torch.equal(output[1], torch.zeros_like(output[1])))
        self.assertTrue(torch.equal(weights[1], torch.zeros_like(weights[1])))

    def test_multihead_projections_shapes_and_divisibility(self):
        module = MultiHeadSelfAttention(model_dim=12, num_heads=3)
        inputs = torch.randn(2, 5, 12)
        self.assertEqual(tuple(module.q_proj(inputs).shape), (2, 5, 12))
        self.assertEqual(tuple(module.k_proj(inputs).shape), (2, 5, 12))
        self.assertEqual(tuple(module.v_proj(inputs).shape), (2, 5, 12))
        mask = torch.tensor([[True, True, False, False, False], [True, True, True, True, True]])
        output, weights = module(inputs, mask)
        self.assertEqual(tuple(output.shape), (2, 5, 12))
        self.assertEqual(tuple(weights.shape), (2, 3, 5, 5))
        self.assertTrue(torch.equal(weights[0, :, :, 2:], torch.zeros_like(weights[0, :, :, 2:])))
        self.assertTrue(torch.equal(output[0, 2:], torch.zeros_like(output[0, 2:])))
        self.assertTrue(torch.allclose(weights[0, :, :2, :2].sum(-1), torch.ones(3, 2)))
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(model_dim=10, num_heads=3)

    def test_feedforward_and_encoder_residual_normalization(self):
        feedforward = PositionwiseFeedForward(8, 16, dropout=0.0)
        inputs = torch.randn(2, 4, 8)
        self.assertEqual(tuple(feedforward(inputs).shape), (2, 4, 8))
        block = TransformerEncoderBlock(8, 2, 16, dropout=0.0)
        mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
        outputs, weights = block(inputs, mask)
        self.assertEqual(tuple(outputs.shape), (2, 4, 8))
        self.assertEqual(tuple(weights.shape), (2, 2, 4, 4))
        self.assertTrue(torch.equal(outputs[~mask], torch.zeros_like(outputs[~mask])))
        self.assertTrue(torch.isfinite(outputs).all())

    def test_classifier_shapes_padding_empty_rows_and_gradients(self):
        torch.manual_seed(42)
        model = CustomTransformerClassifier(
            vocab_size=20,
            model_dim=16,
            num_heads=4,
            num_layers=2,
            feedforward_dim=32,
            dropout=0.0,
            max_length=8,
        )
        ids = torch.tensor([[1, 2, 3, 0], [4, 0, 0, 0], [0, 0, 0, 0]])
        logits, weights = model(ids, return_attention=True)
        self.assertEqual(tuple(logits.shape), (3, 4))
        self.assertEqual(tuple(weights.shape), (3, 4, 4, 4))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.equal(weights[0, :, :, 3], torch.zeros_like(weights[0, :, :, 3])))
        self.assertTrue(torch.equal(weights[1:, :, :, 1:], torch.zeros_like(weights[1:, :, :, 1:])))
        self.assertTrue(torch.equal(weights[2], torch.zeros_like(weights[2])))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2]))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertIsNone(model.position.encoding.grad)

    def test_seeded_model_initialization_is_reproducible(self):
        torch.manual_seed(73)
        first = CustomTransformerClassifier(16, model_dim=8, num_heads=2, max_length=5)
        torch.manual_seed(73)
        second = CustomTransformerClassifier(16, model_dim=8, num_heads=2, max_length=5)
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(first.state_dict().values(), second.state_dict().values())
            )
        )

    def test_invalid_inputs(self):
        model = CustomTransformerClassifier(12, model_dim=8, num_heads=2, max_length=4)
        with self.assertRaises(ValueError):
            model(torch.ones(2, 4, dtype=torch.float32))
        with self.assertRaises(ValueError):
            model(torch.tensor([[1, 0, 2, 0]], dtype=torch.long))
        with self.assertRaises(ValueError):
            model(torch.tensor([[12]], dtype=torch.long))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 0, dtype=torch.long))
        with self.assertRaises(ValueError):
            CustomTransformerClassifier(12, model_dim=8, num_heads=3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward_backward_on_selected_device(self):
        device = torch.device("cuda:0")
        model = CustomTransformerClassifier(32, model_dim=16, num_heads=4, max_length=6).to(device)
        inputs = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], device=device)
        labels = torch.tensor([0, 1], device=device)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        torch.cuda.synchronize(device)
        self.assertEqual(logits.device, device)
        self.assertEqual(next(model.parameters()).device, device)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
