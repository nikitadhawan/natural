import numpy as np
from causality_llm.utils import enumerate_strings, enum_to_dcts

class Estimator:
    # generic class of estimators using LLM-extracted information

    def __init__(self, dataset, name=None):
        self.name = name
        self.dataset = dataset
        
    def get_ate(self, cfg, indices=None):
        self.cfg = cfg
        return None
    
    def get_uncorrected_ate(self, indices=None):
        return None
    
    def get_joint(self, indices=None):
        return None

    def get_propensity_scores(self, indices=None):
        return None
    
    def get_accuracy(self, indices=None):
        return None

    def empirical_joint(self, dataset, data, to_enum=None):
        if to_enum is None:
            _, Y_cols, Z_cols, enum_cols = dataset.XYZ_names(return_enum_cols=True)
            to_enum = enum_cols + [Z_cols, Y_cols]
        options = enumerate_strings(dataset.get_options(to_enum))
        idx_to_feat = enum_to_dcts(options, to_enum)
        idx_to_feat = [dataset.transform_samples(dct) for dct in idx_to_feat]
        data_len = data.shape[0]
        empirical_joint = []
        for features in idx_to_feat:
            subset = data.copy()
            for key in to_enum:
                subset = subset.loc[subset[key] == features["sample_" + key]]
            freq = subset.shape[0] / data_len
            empirical_joint.append(freq)
        return np.array(empirical_joint)
    