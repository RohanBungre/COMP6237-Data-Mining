from sklearn.base import BaseEstimator
import src.main as main

# top_5 = ['NDF', 'US', 'other', 'FR', 'IT']

class BaselineModel(BaseEstimator):

    def __init__(self):
        self.classes_ = ['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT', 'US', 'other']
        pass
        
    
    def fit(self, X, y=None):
        return self


    def transform(self, X):
        return X
    
    def predict_proba(self, X):
         # top_5 = ['NDF', 'US', 'other', 'FR', 'IT'] from notebook
        # ['AU' 'CA' 'DE' 'ES' 'FR' 'GB' 'IT' 'NDF' 'NL' 'PT' 'US' 'other']
        single_pred = [0, 0, 0, 0, 2, 0, 1, 5, 0, 0, 4, 3]
        #print(single_pred * len(X))
        return [single_pred] * len(X)

pipeline = BaselineModel()

main.run_mlflow_pipeline("data/raw", pipeline, run_test=True)
