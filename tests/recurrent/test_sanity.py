import unittest
import torch
from ml.models.recurrent.model import RecurrentClassifier

class SanityTests(unittest.TestCase):
    def test_overfit_tiny_batch(self):
        torch.manual_seed(42);torch.set_num_threads(1)
        x=torch.tensor([[1,1,2],[3,3,4],[5,5,6],[7,7,8]]); y=torch.tensor([0,1,2,3])
        model=RecurrentClassifier('gru',10,8,12,dropout=0);opt=torch.optim.Adam(model.parameters(),lr=.05)
        for _ in range(70):
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();opt.step()
        self.assertTrue(torch.equal(model(x).argmax(1),y))
    def test_attention_masks(self):
        m=RecurrentClassifier('bilstm_attention',10,4,4,dropout=0); x=torch.tensor([[1,2,0,0],[3,0,0,0]])
        _,weights=m(x,True);self.assertTrue(torch.all(weights[x==0]==0));torch.testing.assert_close(weights.sum(1),torch.ones(2))

if __name__=='__main__':unittest.main()
