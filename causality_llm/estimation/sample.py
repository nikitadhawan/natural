import numpy as np
import pandas as pd
from hydra.utils import instantiate

from causality_llm.models import CAUSAL_MODELS
from causality_llm.estimation.estimator import Estimator

class Sample(Estimator):

    def __init__(self, dataset, name="sample"):
        self.name = name
        self.dataset = dataset
        self.observational = True
        self.eval_joint = True
        self.eval_propensity = False
        self.eval_imputation = False
        self.eval_accuracy = True

    def load_data(self, cfg):
        self.samples = pd.read_csv(cfg.data.sample_data_path, index_col=0)
        _, Y_cols, Z_cols, strat_cols = self.dataset.XYZ_names(return_strat_cols=True)
        if cfg.eval:
            val_samples = pd.read_csv(cfg.data.val_sample_data_path, index_col=0)
            self.samples = pd.concat([self.samples, val_samples], ignore_index=True)
        if "inclusion" in cfg.data.sample_data_path:
            self.samples = self.samples.dropna(subset=strat_cols).reset_index(drop=True)
        self.samples = self.samples.head(self.dataset.train_size)
        if cfg.baseline_name not in ["bow", "sentence_encoder"]:
            self.samples = self.dataset.discretize(self.samples, hard_filter=False)

        if cfg.baseline_name == "uniform":
            for col in strat_cols + [Y_cols, Z_cols]:
                options = self.samples[col].unique()
                self.samples[col] = self.samples[col].map(lambda x: np.random.choice(options))
        
    def get_ate(self, indices=None):
        _, Y_cols, Z_cols, strat_cols = self.dataset.XYZ_names(return_strat_cols=True)
        model_name = self.name[7:] # first 7 characters are "sample_"
        Z, Y, X = self.samples[Z_cols].copy(), self.samples[Y_cols].copy(), self.samples[strat_cols].copy()
        if indices is not None:
            data = (X.iloc[indices], X, Z.iloc[indices], Z, Y.iloc[indices], Y) 
        else:
            data = (X, X, Z, Z, Y, Y)
        effect_dict = {}
        model = CAUSAL_MODELS[model_name]()
        train_effect, val_effect = model.get_effect(data) 
        effect_dict["train_" + model_name] = train_effect
        return effect_dict

    def get_uncorrected_ate(self, indices=None):
        _, Y_cols, Z_cols, strat_cols = self.dataset.XYZ_names(return_strat_cols=True)
        model_name = "naive"
        Z, Y, X = self.samples[Z_cols].copy(), self.samples[Y_cols].copy(), self.samples[strat_cols].copy()
        if indices is not None:
            data = (X.iloc[indices], X, Z.iloc[indices], Z, Y.iloc[indices], Y) 
        else:
            data = (X, X, Z, Z, Y, Y)
        effect_dict = {}
        model = CAUSAL_MODELS[model_name]()
        train_effect, val_effect = model.get_effect(data) 
        effect_dict["train_" + self.name[7:]] = train_effect
        return effect_dict

    def get_accuracy(self, cfg, indices=None):
        dataset = instantiate(cfg.data.dataset_class)
        samples_data = self.samples.copy()
        X, Z, Y = dataset.get_data_split(obs=cfg.data.observational)[cfg.data.split]
        if indices is not None:
            samples_data = self.samples.iloc[indices]
            X, Z, Y = X.iloc[indices], Z.iloc[indices], Y.iloc[indices]
        _, Y_cols, Z_cols, self.stratify_cols = self.dataset.XYZ_names(return_strat_cols=True)
        to_eval = self.stratify_cols + [Z_cols, Y_cols]
        X[Y_cols] = Y
        X[Z_cols] = Z   
        acc_metrics = {}
        for field in to_eval:
            true_labels = X[field].to_numpy()
            pred_sample_labels = samples_data[field].to_numpy()
            sample_acc = np.mean(true_labels == pred_sample_labels)
            acc_metrics[field + "_acc_sample"] = sample_acc
        return acc_metrics
