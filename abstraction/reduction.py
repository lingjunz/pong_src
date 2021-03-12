

import os
import joblib
import time
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
import numpy as np

class Reduction(ABC):
    @abstractmethod
    def do_reduction(self, data):
        pass
    

class PCA_R(Reduction):
    def __init__(self, top_components):
        self.top_components = top_components
        self.pca = None
        self.pca_min = None
        self.pca_max = None
        self.explained_variance_ratio = None
        
    def create_pca(self, all_observations):
        assert(len(all_observations) > 0)
        if self.top_components >= all_observations[0].shape[-1]:
            self.pca = None
            return all_observations, np.min(all_observations, axis=0), np.max(all_observations, axis=0)
        else:
            print("build a PCA model...")
            print(">>>original:{}".format(all_observations.shape))
            start_time = time.time()
            self.pca = PCA(n_components=self.top_components)
            self.pca.fit(all_observations)
            pca_data = self.pca.transform(all_observations)
            self.pca_min = np.around(np.min(pca_data,axis=0),4)
            self.pca_max = np.around(np.max(pca_data,axis=0),4)
            self.explained_variance_ratio = self.pca.explained_variance_ratio_
            print(">>>pca_data",pca_data.shape)
            print(">>>explained_variance_ratio",np.sum(self.pca.explained_variance_ratio_))
            print("PCA data min:",self.pca_min)
            print("PCA data max:",self.pca_max)
            print(">>>it takes {} seconds.".format(time.time()-start_time))
            return pca_data, self.pca_min, self.pca_max
        
    def do_reduction(self,data):
        if self.pca is None:
            assert False,"please create a pca model based on training data..."
        else:
            return self.pca.transform(data)
        
        
        

        
        
        