import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import numpy as np
import random
import pandas as pd
import wandb

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from causality_llm.utils import seed_everything, KL, MSE

def save_path_fn(cfg):
    save_path = cfg.data.dataset + "/" + cfg.data.dataset + "_" + cfg.experiment_name 
    if cfg.baseline_name:
        save_path += "_" + cfg.baseline_name
    save_path += "_" + cfg.toeval.estimator.name + "_results"
    save_path = cfg.log_dir + save_path + ".csv"
    return save_path

def eval_estimator(cfg, estimator, obs_estimator, dataset, eval_size=None):
    data_shape = dataset.train_size
    ate_list, naive_list, joint_list, prop_list, acc_list = [], [], [], [], []
    obs_ate_list, obs_joint_list, obs_prop_list = [], [], []
    for i in range(cfg.num_trials):
        np.random.seed(cfg.seed + i)
        estimator.load_data(cfg)
        indices = np.random.choice(data_shape, int(eval_size), replace=False) if cfg.data.observational else None
        ate_list.append(estimator.get_ate(indices))
        naive_list.append(estimator.get_uncorrected_ate(indices))
        if cfg.data.synthetic:
            joint_list.append(estimator.get_joint(indices))
            prop_list.append(estimator.get_propensity_scores(indices))
            acc_list.append(estimator.get_accuracy(cfg, indices))
        enum = True if cfg.toeval.estimator.name == "ipw_enum" else False
        model_name = list(ate_list[0].keys())[0][6:] # first 6 characters of first key will be "train_"
        if cfg.data.synthetic:
            obs_ate_list.append(obs_estimator.get_ate(cfg, indices=indices, model_name=model_name))
            obs_joint_list.append(obs_estimator.get_joint(enum=enum, indices=indices))
            obs_prop_list.append(obs_estimator.get_propensity_scores(enum=enum, indices=indices))

    eval_size = eval_size if eval_size else estimator.dataset.eval_size
    ate_idx = ["ATE " + str(i) for i in range(len(ate_list))]
    result_df = pd.DataFrame(ate_list, index=ate_idx) 
    result_df.loc['ATE_mean'] = result_df.mean()
    result_df.loc['ATE_std_dev'] = result_df.std()

    cre_obs, obs_pred, error_corr, cre_pred = {}, {}, {}, {}
    for model_name in result_df.keys():
        if cfg.data.synthetic:
            obs_arr = np.array([trial[model_name] for trial in obs_ate_list])
            cre_obs_mse = MSE(dataset.ground_truth, obs_arr)
            obs_pred_mse = MSE(obs_arr, result_df.loc[ate_idx, model_name])
            error_correlation = np.mean((dataset.ground_truth - obs_arr)*(obs_arr - result_df.loc[ate_idx, model_name])) * 2
            cre_obs[model_name] = cre_obs_mse
            obs_pred[model_name] = obs_pred_mse
            error_corr[model_name] = error_correlation
        cre_pred_mse = MSE(dataset.ground_truth, result_df.loc[ate_idx, model_name])
        cre_pred[model_name] = cre_pred_mse

    if cfg.data.synthetic:
        result_df.loc["CRE-Obs MSE"] = cre_obs
        result_df.loc["Obs-Pred MSE"] = obs_pred
        result_df.loc["Error Corr"] = error_corr
    result_df.loc["CRE-Pred MSE"] = cre_pred

    # joint distribution
    if estimator.eval_joint and cfg.data.synthetic:
        joint_list = np.array(joint_list)
        obs_joint_list = np.array(obs_joint_list)
        kl_obs_joint = np.array([KL(obs_joint, joint) for (obs_joint, joint) in zip(obs_joint_list, joint_list)]).mean()
        mse_obs_joint = np.array([MSE(obs_joint, joint) for (obs_joint, joint) in zip(obs_joint_list, joint_list)]).mean()
        result_df.loc["KL to Obs Joint"] = kl_obs_joint
        result_df.loc["MSE to Obs Joint"] = mse_obs_joint

    # propensity distribution
    if estimator.eval_propensity and cfg.data.synthetic:
        prop_list = np.array(prop_list)
        obs_prop_list = np.array(obs_prop_list)
        kl_prop = np.array([KL(obs_prop, prop) for (obs_prop, prop) in zip(obs_prop_list, prop_list)]).mean()
        result_df.loc["KL to Obs Propensity"] = kl_prop
        mse_prop = np.array([MSE(obs_prop, prop) for (obs_prop, prop) in zip(obs_prop_list, prop_list)]).mean()
        result_df.loc["MSE to Obs Propensity"] = mse_prop

    if estimator.eval_accuracy and cfg.data.synthetic:
        for key in acc_list[0].keys():
            result_df.loc[key + "_mean"] = np.array([acc_dct[key] for acc_dct in acc_list]).mean()
            result_df.loc[key + "_std"] = np.array([acc_dct[key] for acc_dct in acc_list]).std()
        
    column_names = [col + "_eval" + str(eval_size) for col in result_df.columns]
    result_df = result_df.rename(columns=dict(zip(result_df.columns, column_names)))
    return result_df


@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    seed_everything(cfg.seed) 

    dataset = instantiate(cfg.data.dataset_class)
    obs_estimator = instantiate(cfg.obs.estimator) if cfg.data.synthetic else None
    estimator = instantiate(cfg.toeval.estimator)

    all_results = pd.DataFrame()
    eval_sizes = cfg.data.eval_size if cfg.data.eval_size else [cfg.data.eval_size]
    for eval_size in eval_sizes:
        result_df = eval_estimator(cfg, estimator, obs_estimator, dataset, eval_size=eval_size)
        all_results = all_results.merge(result_df, how="outer", left_index=True, right_index=True)

    save_path = save_path_fn(cfg)

    print("Saving to: ", save_path)
    all_results.to_csv(save_path)

if __name__ == "__main__":
    main()