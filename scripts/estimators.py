import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
class CustomNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.priors = []
        self.loglikelihoods = []
               
    def fit(self, X, y):
        
        X = X.todense()
        X = np.array(X)
        y = np.array(y)
        #np.log
        self.priors = np.log([
            np.sum(y == 0) / len(y),  
            np.sum(y == 1) / len(y)   
        ])
        zeros = np.array(np.sum(X[np.where(y == 0)], axis=0) + 1)
        ones = np.array(np.sum(X[np.where(y == 1)], axis=0) + 1)
        self.loglikelihoods = np.log([(zeros / (np.sum(zeros))), (ones / (np.sum(ones)))])
        
        return self
        

    def predict(self, X):
        return np.array(np.argmax(np.add((X @ np.transpose(self.loglikelihoods)), self.priors), axis=1))
        
        
     
    
