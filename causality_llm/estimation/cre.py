import numpy as np

from hydra.utils import instantiate

from causality_llm.utils import enumerate_strings, enum_to_dcts
from causality_llm.models import CAUSAL_MODELS
from causality_llm.estimation.estimator import Estimator

class CRE(Estimator):

    def __init__(self, dataset, name="cre"):
        self.name = name
        self.dataset = dataset
        self.observational = False
        self.eval_joint = False
        self.eval_propensity = False
        self.eval_imputation = False
        self.eval_accuracy = False

    def get_ate(self, cfg, indices=None):
        self.cfg = cfg
        self.cfg.data.observational = self.observational
        dataset = instantiate(cfg.data.dataset_class)
        X_cols, _, _ = dataset.XYZ_names()
        data_dict = dataset.get_data_split(obs=cfg.data.observational)
        X_train, Z_train, Y_train = data_dict["train"]
        X_val, Z_val, Y_val = data_dict["val"]
        X_train, X_val = X_train[X_cols], X_val[X_cols]
        if indices is not None:
            X_train, Z_train, Y_train = X_train.iloc[indices], Z_train.iloc[indices], Y_train.iloc[indices]

        data = (X_train, X_val, Z_train, Z_val, Y_train, Y_val)
        effect_dict = {}
        for model_name in CAUSAL_MODELS.keys():
            model = CAUSAL_MODELS[model_name]()
            train_effect, val_effect = model.get_effect(data)
            effect_dict["train_" + model_name] = train_effect
            effect_dict["val_" + model_name] = val_effect
            
        return effect_dict
    
    def get_joint(self, enum=True, indices=None):
        if enum:
            _, Y_cols, Z_cols, cols = self.dataset.XYZ_names(return_enum_cols=True)
        else:
            _, Y_cols, Z_cols, cols = self.dataset.XYZ_names(return_strat_cols=True)
        X, Z, Y = self.dataset.get_data_split(obs=self.observational)["train"]
        data = X.copy()[cols] if indices is None else X.iloc[indices].copy()[cols] 
        data[Y_cols] = Y if indices is None else Y.iloc[indices]
        data[Z_cols] = Z if indices is None else Z.iloc[indices]
        return self.empirical_joint(self.dataset, data)

    def get_propensity_scores(self, enum=True, indices=None):
        if enum:
            _, Y_cols, Z_cols, cols = self.dataset.XYZ_names(return_enum_cols=True)
        else:
            _, Y_cols, Z_cols, cols = self.dataset.XYZ_names(return_strat_cols=True)
        X, Z, Y = self.dataset.get_data_split(obs=self.observational)["train"]
        data = X.copy()[cols] if indices is None else X.iloc[indices].copy()[cols] 
        data[Y_cols] = Y if indices is None else Y.iloc[indices]
        data[Z_cols] = Z if indices is None else Z.iloc[indices]
        data = self.dataset.discretize(data)
        options = enumerate_strings(self.dataset.get_options(cols))
        idx_to_feat = enum_to_dcts(options, cols)
        idx_to_feat = [self.dataset.transform_samples(dct) for dct in idx_to_feat]
        
        prop_score_lst = []
        for features in idx_to_feat:
            subset = data.copy()
            # restrict data using features
            for key in cols:
                subset = subset.loc[subset[key] == features["sample_" + key]]
            # marginalize out Y
            control = len(subset.loc[subset[Z_cols] == 0])
            treat = len(subset.loc[subset[Z_cols] == 1])
            if control + treat == 0:
                prop_scores = [0, 0]
            else:
                prop_scores = [control / (control + treat), treat / (control + treat)]
            prop_score_lst.append(prop_scores)
        return prop_score_lst
    
    def get_accuracy(self, cfg, indices=None):
        return None
