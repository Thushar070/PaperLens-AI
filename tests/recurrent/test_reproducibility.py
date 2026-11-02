import unittest
import torch
from ml.models.recurrent.model import RecurrentClassifier

class ReproducibilityTests(unittest.TestCase):
    def test_seeded_initialization(self):
        x=torch.tensor([[1,2,0]])
        torch.manual_seed(42); a=RecurrentClassifier('lstm',10,4,4,dropout=0)
        torch.manual_seed(42); b=RecurrentClassifier('lstm',10,4,4,dropout=0)
        torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
