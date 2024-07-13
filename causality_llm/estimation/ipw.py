import numpy as np
import pandas as pd
from hydra.utils import instantiate

from causality_llm.utils import enumerate_strings, enum_to_dcts
from causality_llm.estimation.estimator import Estimator
    
class IPW(Estimator):
    
    def __init__(self, dataset, name="ipw_enum"):
        self.name = name
        self.dataset = dataset
        _, Y_cols, Z_cols, self.enum_cols = dataset.XYZ_names(return_enum_cols=True)
        to_enum = self.enum_cols + [Z_cols, Y_cols] 
        self.outcome_axis = len(to_enum) - 1
        options = dataset.get_options(to_enum)
        self.joint_shape = [len(options[field]) for field in to_enum] # hillstrom: [2, 2, 2, 2]
        self.treatment_dim = len(options[Z_cols]) # hillstrom: 2
        self.outcome_dim = len(options[Y_cols]) # hillstrom: 2
        self.feature_dim = np.prod(np.array(self.joint_shape)) // (self.treatment_dim * self.outcome_dim) # hillstrom: 4
        self.prop_score_lst = None
        self.eval_joint = True
        self.eval_propensity = True
        self.eval_accuracy = True

    def compute_prop_score(self, data, dataset):
        options = enumerate_strings(dataset.get_options(self.enum_cols))
        idx_to_feat = enum_to_dcts(options, self.enum_cols)
        feat_dicts = [dataset.transform_samples(dct) for dct in idx_to_feat] # len(feat_dict) = self.feature_dim
        prop_score_lst = []
        
        for i in range(len(feat_dicts)):
            subset = data.copy()
            # marginalize out Y
            propensity = subset[["probs"]].apply(lambda row: np.sum(row["probs"], axis=self.outcome_axis).reshape((-1, self.treatment_dim)), axis=1) # [self.feature_dim, self.treatment_dim]
            # get P(T=1, X=x | R)
            treat_numerator = propensity.apply(lambda arr: arr[i][1]) 
            control_numerator = propensity.apply(lambda arr: arr[i][0]) 
            # get P(X=x | R)
            denominator = propensity.apply(lambda arr: arr[i].sum()) 
            # average over posts
            control = control_numerator.mean() / denominator.mean()
            treat = treat_numerator.mean() / denominator.mean()
            prop_scores = [control, treat]
            prop_score_lst.append(prop_scores)
        return prop_score_lst

    def load_data(self, cfg):
        self.probs = pd.read_csv(cfg.data.probs_data_path, index_col=0)
        if cfg.eval:
            val_probs = pd.read_csv(cfg.data.val_probs_data_path, index_col=0)
            self.probs = pd.concat([self.probs, val_probs], ignore_index=True)
        self.probs = self.probs.head(self.dataset.train_size)
    
    def get_ate(self, indices=None):
        data = self.probs.copy()
        if indices is not None:
            data = data.iloc[indices]
        # reshape data["probs"] to have self.joint_shape, where axes 0:-2 correspond to X, axis -2 corresponds to Z, axis -1 correspond to Y
        data.loc[:, "probs"] = data.apply(lambda row: np.array([float(prob) for prob in row["probs"][1:-1].split()]).reshape(self.joint_shape), axis=1)
        self.prop_score_lst = self.compute_prop_score(data, self.dataset)

        treat, control = 0, 0

        for row in data.iterrows():
            row = row[1]
            probs = row["probs"]
            # enumerate treatments
            for z in range(self.treatment_dim):
                # enumerate outcomes
                for y in range(self.outcome_dim):
                    # enumerate features
                    for i in range(len(self.prop_score_lst)):
                        # joint probability
                        probs = probs.reshape((self.feature_dim, self.treatment_dim, self.outcome_dim))
                        # probability of this enumerated possibility
                        posterior = probs[i, z, y]
                        # propensity score given these features
                        z_given_x = self.prop_score_lst[i]
                        # ignore propensity sores of 0
                        if 0 not in z_given_x:                           
                            treat += (z * y * posterior / z_given_x[1])  
                            control += ((1 - z) * y * posterior / z_given_x[0])

        prob_ate = (treat - control) / len(data)
        return {"train_ipw": prob_ate}
    
    def get_uncorrected_ate(self, indices=None):
        data = self.probs.copy()
        if indices is not None:
            data = data.iloc[indices]
        # reshape data["probs"] to have self.joint_shape, where axes 0:-2 correspond to X, axis -2 corresponds to Z, axis -1 correspond to Y
        data.loc[:, "probs"] = data.apply(lambda row: np.array([float(prob) for prob in row["probs"][1:-1].split()]).reshape(self.joint_shape), axis=1)

        treat, control = 0, 0

        for row in data.iterrows():
            row = row[1]
            probs = row["probs"]
            # enumerate treatments
            for z in range(self.treatment_dim):
                # enumerate outcomes
                for y in range(self.outcome_dim):
                    # enumerate features
                    for i in range(len(self.prop_score_lst)):
                        # joint probability
                        probs = probs.reshape((self.feature_dim, self.treatment_dim, self.outcome_dim))
                        # probability of this enumerated possibility
                        posterior = probs[i, z, y]                    
                        treat += (z * y * posterior)  
                        control += ((1 - z) * y * posterior)
                            
        prob_ate = (treat - control) / len(data)
        return {"train_ipw": prob_ate}
    
    def get_joint(self, indices=None, compare="obs"):
        probs_data = self.probs.copy()
        if indices is not None:
            probs_data = probs_data.iloc[indices]
        if compare == "cre":
            pred_joint = np.zeros((self.feature_dim, self.treatment_dim, self.outcome_dim))
            pred_joint_denom = np.zeros((self.feature_dim, self.treatment_dim, self.outcome_dim))
            for row in probs_data.iterrows():
                row = row[1]
                probs = np.array([float(prob) for prob in row["probs"][1:-1].split()]).reshape(self.joint_shape)
                for z in range(self.treatment_dim):
                    for y in range(self.outcome_dim):
                        for i in range(len(self.prop_score_lst)):
                            probs = probs.reshape((self.feature_dim, self.treatment_dim, self.outcome_dim))
                            posterior = probs[i, z, y]
                            z_given_x = self.prop_score_lst[i]
                            if 0 not in z_given_x:
                                if z == 0:
                                    pred_joint[i, z, y] += (1 - z) * posterior / z_given_x[0]
                                    pred_joint_denom[i, z, y] += (1 - z) / z_given_x[0] 
                                else:
                                    pred_joint[i, z, y] += z * posterior / z_given_x[1] 
                                    pred_joint_denom[i, z, y] += z / z_given_x[1]
                                    
            pred_joint = pred_joint / pred_joint_denom
            pred_joint = pred_joint.flatten()
        else: # compare = "obs"
            probs_per_post = probs_data.apply(lambda row: [float(prob) for prob in row["probs"][1:-1].split()], axis=1) # [np.prod(self.joint_shape)]
            probs_per_post = np.array(probs_per_post.values.tolist())
            pred_joint = np.mean(probs_per_post, axis=0)
        return pred_joint

    def get_propensity_scores(self, indices=None):
        return self.prop_score_lst
    
    def get_accuracy(self, cfg, indices=None):
        dataset = instantiate(cfg.data.dataset_class)
        X, Z, Y = dataset.get_data_split(obs=cfg.data.observational)[cfg.data.split]
        probs = self.probs.copy()
        if indices is not None:
            probs = probs.iloc[indices]
            X, Z, Y = X.iloc[indices], Z.iloc[indices], Y.iloc[indices]
        _, Y_cols, Z_cols, self.enum_cols = self.dataset.XYZ_names(return_enum_cols=True)
        to_eval = self.enum_cols + [Z_cols, Y_cols]
        X[Y_cols] = Y
        X[Z_cols] = Z
        acc_metrics = {}
        for field in to_eval:
            true_labels = X[field].to_numpy()
            pred_sample_labels = probs["sample_" + field].to_numpy()
            pred_max_labels = probs["max_sample_" + field].to_numpy()
            sample_acc = np.mean(true_labels == pred_sample_labels)
            max_acc = np.mean(true_labels == pred_max_labels)
            acc_metrics[field + "_acc_sample"] = sample_acc
            acc_metrics[field + "_acc_max"] = max_acc
        return acc_metrics
