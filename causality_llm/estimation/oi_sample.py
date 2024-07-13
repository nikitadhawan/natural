import numpy as np
import pandas as pd
from hydra.utils import instantiate

from causality_llm.utils import enumerate_strings, enum_to_dcts
from causality_llm.estimation.estimator import Estimator
    
class OISample(Estimator):
    
    def __init__(self, dataset, name="oi_sample"):
        self.name = name
        self.dataset = dataset
        _, self.Y_cols, self.Z_cols, self.stratify_cols = dataset.XYZ_names(return_strat_cols=True)
        to_enum = [self.Y_cols] 
        self.outcome_axis = len(to_enum) - 1
        options = dataset.get_options(to_enum)
        self.joint_shape = [len(options[field]) for field in to_enum] # hillstrom: [2]
        self.outcome_dim = len(options[self.Y_cols]) # hillstrom: 2
        self.eval_joint = False
        self.eval_propensity = False
        self.eval_imputation = True
        self.eval_accuracy = True

    def compute_outcome_dist(self, data, dataset):
        options = enumerate_strings(dataset.get_options(self.stratify_cols))
        idx_to_feat = enum_to_dcts(options, self.stratify_cols)
        feat_dicts = [dataset.transform_samples(dct) for dct in idx_to_feat] 
        outcome_dist_lst, ratio_x = [], []
        n = len(data)
        
        for i in range(len(feat_dicts)):
            features = feat_dicts[i]
            subset = data.copy()
            # restrict posts using sampled features
            for key in self.stratify_cols:
                subset = subset.loc[subset["sample_" + key] == features["sample_" + key]]
            # get posts in control and treat 
            control = subset.loc[subset["sample_" + self.Z_cols] == 0]
            treat = subset.loc[subset["sample_" + self.Z_cols] == 1]
            # get p(Y=y | T, X)
            control_y0 = np.array([prob[0] for prob in control["probs"]]) 
            treat_y0 = np.array([prob[0] for prob in treat["probs"]]) 
            control_y1 = np.array([prob[1] for prob in control["probs"]]) 
            treat_y1 = np.array([prob[1] for prob in treat["probs"]])
            # average over posts
            if len(control) > 0 and len(treat) > 0:
                control_y0_cond = np.mean(control_y0)
                treat_y0_cond = np.mean(treat_y0)
                control_y1_cond = np.mean(control_y1)
                treat_y1_cond = np.mean(treat_y1)
                outcome_dist = [control_y0_cond, control_y1_cond, treat_y0_cond, treat_y1_cond]
                outcome_dist_lst.append(outcome_dist)
                ratio_x.append([len(control_y0)/n, len(control_y1)/n, len(treat_y0)/n, len(treat_y1)/n])
        return np.array(outcome_dist_lst), np.array(ratio_x)

    def load_data(self, cfg):
        self.probs = pd.read_csv(cfg.data.probs_data_path, index_col=0)
        self.samples = pd.read_csv(cfg.data.sample_data_path, index_col=0)
        if cfg.eval:
            val_probs = pd.read_csv(cfg.data.val_probs_data_path, index_col=0)
            self.probs = pd.concat([self.probs, val_probs], ignore_index=True)
            val_samples = pd.read_csv(cfg.data.val_sample_data_path, index_col=0)
            self.samples = pd.concat([self.samples, val_samples], ignore_index=True)
        self.samples = self.dataset.discretize(self.samples, hard_filter=False)
        if "inclusion" in cfg.data.sample_data_path:
            self.samples = self.samples.dropna(subset=self.stratify_cols).reset_index(drop=True)
        self.probs[["sample_" + x_col for x_col in self.stratify_cols]] = self.samples[self.stratify_cols].copy()
        self.probs["sample_" + self.Z_cols] = self.samples[self.Z_cols].copy()
        self.probs = self.probs.head(self.dataset.train_size)

    def get_ate(self, indices=None):
        probs_data = self.probs.copy()
        if indices is not None:
            probs_data = probs_data.iloc[indices]
        # reshape data["probs"] to have self.joint_shape, where axis 0 corresponds to Y
        probs_data.loc[:, "probs"] = probs_data.apply(lambda row: np.array([float(prob) for prob in row["probs"][1:-1].split()]).reshape(self.joint_shape), axis=1)
        
        self.outcome_dist_lst, self.ratio_x = self.compute_outcome_dist(probs_data, self.dataset)
        # marginalize out X 
        self.outcome_dist_lst = np.average(self.outcome_dist_lst, axis=0, weights=self.ratio_x)
        # we only need y1 conditionals since Y is binary
        _, control_y1_cond, _, treat_y1_cond = self.outcome_dist_lst
        oi_ate = treat_y1_cond - control_y1_cond
        return {"train_standardization": oi_ate}
    
    def get_accuracy(self, cfg, indices=None):
        dataset = instantiate(cfg.data.dataset_class)
        X, Z, Y = dataset.get_data_split(obs=cfg.data.observational)[cfg.data.split]
        probs = self.probs.copy()
        if indices is not None:
            probs = probs.iloc[indices]
            X, Z, Y = X.iloc[indices], Z.iloc[indices], Y.iloc[indices]
        _, Y_cols, Z_cols, self.stratify_cols = self.dataset.XYZ_names(return_strat_cols=True)
        to_eval = self.stratify_cols + [Z_cols, Y_cols]
        X[Y_cols] = Y
        X[Z_cols] = Z
        acc_metrics = {}
        for field in to_eval:
            true_labels = X[field].to_numpy()
            pred_sample_labels = probs["sample_" + field].to_numpy()
            sample_acc = np.mean(true_labels == pred_sample_labels)
            acc_metrics[field + "_acc_sample"] = sample_acc
        return acc_metrics
