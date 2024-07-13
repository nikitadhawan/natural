import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from causality_llm.models import LLAMA_CKPT, llama, chat_llama, chat_llama3
from causality_llm.utils import seed_everything, qa_interleaved_enum, enumerate_strings, enum_to_dcts, get_sample_text

import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', FutureWarning)


def save_path_fn(cfg, final=False):
    save_path = cfg.save_path + cfg.data.dataset + "/"
    if final:
        save_path += cfg.date_time + "_" + os.environ["SLURM_JOBID"] + "_"
    save_path += cfg.data.dataset + "_" + cfg.model.llm_name + "_" + cfg.prompt_type + "_" 
    save_path += cfg.experiment_name + "_probs.csv"
    return save_path

def get_model(cfg):
    ckpt_dir = LLAMA_CKPT[cfg.model.llm_name]
    tokenizer_path = LLAMA_CKPT["llama2_tokenizer"]
    model = chat_llama if cfg.model.llm_name[-4:] == "chat" else llama
    if "llama3" in cfg.model.llm_name:
        model = chat_llama3
        tokenizer_path = LLAMA_CKPT[cfg.model.llm_name + "_tokenizer"]
    llm = model(ckpt_dir=ckpt_dir,
                tokenizer_path=tokenizer_path,
                temperature=cfg.model.temperature,
                max_batch_size=2,
                max_gen_len=1,
                seed=cfg.seed)

    prompt_file = open(cfg.prompt_dir + cfg.data.dataset + "_" + cfg.prompt_type + ".txt", "r")
    system_template = prompt_file.read()
    llm.system_template = system_template
    return llm

@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    dataset = instantiate(cfg.data.dataset_class)
    llm = get_model(cfg)
    llm_posts_df = pd.read_csv(cfg.data.posts_path, index_col=0).head(cfg.data.dataset_size)
    
    X_cols, Y_cols, Z_cols, enum_cols = dataset.XYZ_names(return_enum_cols=True)
    _, _, _, strat_cols = dataset.XYZ_names(return_strat_cols=True)
    llm_probs_df = pd.DataFrame()
    save_path = save_path_fn(cfg)
    
    if not cfg.enum_x:
        llm_samples_df = pd.read_csv(cfg.data.samples_path, index_col=0)
        if "inclusion" in cfg.data.samples_path:
            llm_posts_df = dataset.discretize(llm_posts_df, hard_filter=False)
            llm_samples_df = dataset.discretize(llm_samples_df, hard_filter=False)
            X_cols = strat_cols
        else:
            llm_samples_df = dataset.discretize(llm_samples_df)
        llm_samples_df = llm_samples_df.dropna(subset=strat_cols)

    print(llm_posts_df)
    if cfg.restore:
        llm_probs_df = pd.read_csv(save_path, index_col=0)
        llm_posts_df = llm_posts_df.iloc[len(llm_probs_df):]

    if cfg.enum_x:
        to_enum = enum_cols + [Z_cols, Y_cols] 
    elif cfg.use_samples == "ipw":
        to_enum = [Z_cols, Y_cols]
    else: # cfg.use_samples = oi
        to_enum = [Y_cols] # OI needs only [Y_cols]
    
    options = enumerate_strings(dataset.get_options(to_enum))
    interleaved_options = qa_interleaved_enum(
        dataset.get_question_prompt(to_enum),
        dataset.get_options(to_enum),
        options,
        to_enum
    )
    idx_to_feat = enum_to_dcts(options, to_enum)
    idx_to_feat = [dataset.transform_samples(dct) for dct in idx_to_feat]
    llm_inputs, rows = [], []
    for i, row in tqdm(llm_posts_df.iterrows()):
        X = row["post"]
        if not cfg.enum_x:
            # get corresponding row from llm_samples_df
            sample_row = llm_samples_df.loc[llm_samples_df["post"] == X]
            if len(sample_row) == 0:
                continue;
            if cfg.use_samples == "oi":
                strat_cols += [Z_cols]
            sample_row = sample_row[strat_cols]
            sample_row = sample_row.to_dict('records')[0]
            sample_text = get_sample_text(sample_row, dataset)
            X += sample_text
        llm_inputs.append(X)
        if i == 0:
            print(X)
        rows.append(row[X_cols + [Y_cols, Z_cols, "post"]])
        if len(llm_inputs) >= cfg.model.batch_size:
            post_probs, sample_indices, max_indices = llm.compute_input_probs(llm_inputs, interleaved_options)
            dict_to_save = [{**rows[j].to_dict(), 
                             **idx_to_feat[sample_indices[j]], 
                             **{"max_" + k: v for k, v in idx_to_feat[max_indices[j]].items()},
                             **{"probs": post_probs[j], 
                                "posterior": post_probs[j][max_indices[j]]}} for j in range(len(llm_inputs))]
            df_to_save = pd.DataFrame.from_dict(dict_to_save)
            llm_probs_df = pd.concat([llm_probs_df, df_to_save], ignore_index=True)
            llm_probs_df.to_csv(save_path)
            llm_inputs, rows = [], []

    save_path_final = save_path_fn(cfg, final=True)
    llm_probs_df.to_csv(save_path_final)

if __name__ == "__main__":
    main()