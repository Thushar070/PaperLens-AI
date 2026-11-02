import unittest
import torch
from ml.models.recurrent.model import RecurrentClassifier

class RecurrentTests(unittest.TestCase):
    def test_models_padding_gradients(self):
        for name in ('rnn','lstm','gru','bilstm','bilstm_attention'):
            torch.manual_seed(42); m=RecurrentClassifier(name,20,8,6); m.eval()
            x=torch.tensor([[1,2,3,0,0],[4,0,0,0,0],[0,0,0,0,0]])
            y,attention=m(x,True); self.assertEqual(tuple(y.shape),(3,4)); self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(torch.allclose(y[:2],m(x[:2,:3]),atol=1e-6))
            y.sum().backward(); self.assertIsNotNone(m.embedding.weight.grad)
            if attention is not None:
                self.assertTrue(torch.all(attention[x==0]==0)); self.assertAlmostEqual(attention[0].sum().item(),1,places=6)
    def test_invalid_padding(self):
        m=RecurrentClassifier('gru',10,4,4)
        with self.assertRaises(ValueError): m(torch.tensor([[1,0,2]]))
        with self.assertRaises(ValueError): RecurrentClassifier('bad',10)

if __name__=='__main__': unittest.main()
