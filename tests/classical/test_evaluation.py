import unittest
from sklearn.naive_bayes import MultinomialNB
from ml.models.classical.training import evaluate
from ml.models.classical.inference import predict

class EvaluationTests(unittest.TestCase):
    def test_metrics_and_order(self):
        model=MultinomialNB().fit([[2,0],[0,2]],[0,1])
        metrics,pred=evaluate(model,[[2,0],[0,2]],[0,1],['first','second'])
        self.assertEqual(metrics['accuracy'],1)
        self.assertEqual(metrics['confusion_matrix'],[[1,0],[0,1]])
        self.assertEqual(list(pred),[0,1])
        self.assertEqual(metrics['report']['first']['f1-score'],1)

    def test_inference_invalid_inputs(self):
        with self.assertRaises(ValueError): predict('title','abstract','bad')
        with self.assertRaises(ValueError): predict(' ',' ','bow_nb')
        with self.assertRaises(ValueError): predict(None,'a','bow_nb')

if __name__=='__main__': unittest.main()
