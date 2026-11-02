import tempfile,unittest
from pathlib import Path
import torch
from ml.models.recurrent.model import RecurrentClassifier
from ml.models.recurrent.train import batches,run_epoch,atomic_torch

class TrainingTests(unittest.TestCase):
    def test_train_and_reload(self):
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model=RecurrentClassifier('gru',12,4,4).to(device)
        x=torch.tensor([[1,2,0],[2,3,4],[4,0,0],[5,3,0]]);y=torch.tensor([0,1,2,3]); opt=torch.optim.Adam(model.parameters(),lr=.01)
        before=model.head.weight.detach().clone(); result=run_epoch(model,batches(x,y,2),device,opt)
        self.assertGreater(result['loss'],0);self.assertFalse(torch.equal(before,model.head.weight))
        model.eval(); expected=model(x.to(device)).detach().cpu()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';atomic_torch(path,model.state_dict());copy=RecurrentClassifier('gru',12,4,4).to(device);copy.load_state_dict(torch.load(path,map_location=device,weights_only=True));copy.eval()
            torch.testing.assert_close(expected,copy(x.to(device)).detach().cpu())

if __name__=='__main__': unittest.main()
