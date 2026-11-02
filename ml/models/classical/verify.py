"""Verify saved model inference and immutable inputs."""
import hashlib,json,platform
from pathlib import Path
import sklearn,numpy,scipy,gensim
from ml.models.classical.training import EXPERIMENTS
from ml.models.classical.inference import predict
from ml.models.classical.run import CHECKPOINT,OUT,save

def main():
    checkpoint=json.loads(CHECKPOINT.read_text(encoding='utf-8'))
    integrity={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected for name,expected in checkpoint['input_sha256'].items()}
    if not all(integrity.values()): raise RuntimeError('Earlier artifact integrity failure')
    results={name:predict('A new graph learning method','We study neural approaches to graph classification.',name) for name in EXPERIMENTS}
    save(OUT/'verification.json',{'input_integrity':integrity,'inference':results,'versions':{'python':platform.python_version(),'scikit_learn':sklearn.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'gensim':gensim.__version__}})
    print('Verified',len(integrity),'input hashes and',len(results),'inference paths')

if __name__=='__main__':main()
