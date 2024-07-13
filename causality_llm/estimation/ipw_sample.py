import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from hydra.utils import instantiate

from causality_llm.utils import enumerate_strings, enum_to_dcts
from causality_llm.estimation.estimator import Estimator
    
class IPWSample(Estimator):
    
    def __init__(self, dataset, name="ipw_sample"):
        self.name = name
        self.dataset = dataset
        _, self.Y_cols, self.Z_cols, self.stratify_cols = dataset.XYZ_names(return_strat_cols=True)
        to_enum = [self.Z_cols, self.Y_cols] 
        self.outcome_axis = len(to_enum) - 1
        options = dataset.get_options(to_enum)
        self.joint_shape = [len(options[field]) for field in to_enum] # hillstrom: [2, 2]
        self.treatment_dim = len(options[self.Z_cols]) # hillstrom: 2
        self.outcome_dim = len(options[self.Y_cols]) # hillstrom: 2
        self.prop_score_lst = None
        self.eval_joint = False
        self.eval_propensity = True
        self.eval_accuracy = True

    def compute_prop_score(self, data, dataset):
        options = enumerate_strings(dataset.get_options(self.stratify_cols))
        idx_to_feat = enum_to_dcts(options, self.stratify_cols)
        feat_dicts = [dataset.transform_samples(dct) for dct in idx_to_feat] 
        prop_score_lst = []
        
        for i in range(len(feat_dicts)):
            features = feat_dicts[i]
            subset = data.copy()
            # restrict posts using sampled features
            for key in self.stratify_cols:
                subset = subset.loc[subset["sample_" + key] == features["sample_" + key]]
            if len(subset) == 0:
                prop_scores = [0, 0]
            else:
                # marginalize out Y
                propensity = subset[["probs"]].apply(lambda row: np.sum(row["probs"], axis=self.outcome_axis), axis=1) # [self.treatment_dim]
                # average over posts
                control = propensity.apply(lambda arr: arr[0]).sum() / len(subset)
                treat = propensity.apply(lambda arr: arr[1]).sum() / len(subset)
                prop_scores = [control, treat]
            prop_score_lst.append(prop_scores)
        return prop_score_lst

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
        self.probs = self.probs.head(self.dataset.train_size)

    def get_ate(self, indices=None):
        probs_data = self.probs.copy()
        options = enumerate_strings(self.dataset.get_options(self.stratify_cols))
        idx_to_feat = enum_to_dcts(options, self.stratify_cols)
        feat_dicts = [self.dataset.transform_samples(dct) for dct in idx_to_feat]

        if indices is not None:
            probs_data = probs_data.iloc[indices]
        # reshape data["probs"] to have self.joint_shape, where axis 0 corresponds to Z, axis 1 corresponds to Y
        probs_data.loc[:, "probs"] = probs_data.apply(lambda row: np.array([float(prob) for prob in row["probs"][1:-1].split()]).reshape(self.joint_shape), axis=1)
        
        self.prop_score_lst = self.compute_prop_score(probs_data, self.dataset)

        treat, control = 0, 0

        for row in probs_data.iterrows():
            row = row[1]
            probs = row["probs"]
            sampled_cols = ["sample_" + c for c in self.stratify_cols]
            x = row[sampled_cols].to_dict()
            # enumerate treatments
            for z in range(self.treatment_dim):
                # enumerate outcomes
                for y in range(self.outcome_dim):
                    # probability of this enumerated possibility
                    posterior = probs[z, y]
                    # propensity score given x features
                    i = feat_dicts.index(x)
                    z_given_x = self.prop_score_lst[i]
                    # ignore propensity sores of 0
                    if 0 not in z_given_x:                           
                        treat += (z * y * posterior / z_given_x[1])  
                        control += ((1 - z) * y * posterior / z_given_x[0])

        prob_ate = (treat - control) / len(probs_data)
        return {"train_ipw": prob_ate}

    def check_balance_xi(self, covariate, indices=None):
        probs_data = self.probs.copy()
        options = enumerate_strings(self.dataset.get_options(self.stratify_cols))
        idx_to_feat = enum_to_dcts(options, self.stratify_cols)
        feat_dicts = [self.dataset.transform_samples(dct) for dct in idx_to_feat]

        if indices is not None:
            probs_data = probs_data.iloc[indices]
        # reshape data["probs"] to have self.joint_shape, where axis 0 corresponds to Z, axis 1 corresponds to Y
        probs_data.loc[:, "probs"] = probs_data.apply(lambda row: np.array([float(prob) for prob in row["probs"][1:-1].split()]).reshape(self.joint_shape), axis=1)

        treat, control = [], []

        for row in probs_data.iterrows():
            row = row[1]
            sampled_cols = ["sample_" + c for c in self.stratify_cols]
            x = row[sampled_cols].to_dict()
            xi = row["sample_" + covariate]
            t = row["sample_" + self.Z_cols]
            i = feat_dicts.index(x)
            z_given_x = self.prop_score_lst[i]
            if 0 not in z_given_x:                           
                treat += [t * xi / z_given_x[1]]  
                control += [(1 - t) * xi / z_given_x[0]]

        treat, control = np.array(treat), np.array(control)
        denom = np.sqrt(0.5 * (np.var(treat) + np.var(control)))
        balance = (np.mean(treat) - np.mean(control)) / denom
        return balance
    
    def check_balance(self, indices=None):
        covariate_balance = {}
        for covariate in self.stratify_cols:
            covariate_balance[covariate] = self.check_balance_xi(covariate, indices=indices)
        return covariate_balance


    def get_uncorrected_ate(self, indices=None):
        probs_data = self.probs.copy()
        options = enumerate_strings(self.dataset.get_options(self.stratify_cols))
        idx_to_feat = enum_to_dcts(options, self.stratify_cols)
        feat_dicts = [self.dataset.transform_samples(dct) for dct in idx_to_feat]

        if indices is not None:
            probs_data = probs_data.iloc[indices]
        # reshape data["probs"] to have self.joint_shape, where axis 0 corresponds to Z, axis 1 corresponds to Y
        probs_data.loc[:, "probs"] = probs_data.apply(lambda row: np.array([float(prob) for prob in row["probs"][1:-1].split()]).reshape(self.joint_shape), axis=1)
    
        treat, control = 0, 0

        for row in probs_data.iterrows():
            row = row[1]
            probs = row["probs"]
            sampled_cols = ["sample_" + c for c in self.stratify_cols]
            x = row[sampled_cols].to_dict()
            # enumerate treatments
            for z in range(self.treatment_dim):
                # enumerate outcomes
                for y in range(self.outcome_dim):
                    # probability of this enumerated possibility
                    posterior = probs[z, y]                       
                    treat += (z * y * posterior)  
                    control += ((1 - z) * y * posterior)
    
        prob_ate = (treat - control) / len(probs_data)
        return {"train_ipw": prob_ate}

    def get_propensity_scores(self, indices=None):
        return self.prop_score_lst
    
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
