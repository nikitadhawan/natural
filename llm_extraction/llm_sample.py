import os
import numpy as np
import json
import pandas as pd
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import nest_asyncio

from causality_llm.models import direct_gpt
from causality_llm.utils import seed_everything


def save_path_fn(cfg, final=False):
    save_path = cfg.save_path + cfg.data.dataset + "/"
    if final:
        save_path += cfg.date_time + "_"
    save_path += cfg.prompt_type + "_" + cfg.data.dataset + "_" + cfg.experiment_name + "_" + cfg.model.llm_name 
    save_path += "_samples.csv"
    return save_path

def get_model(cfg):
    prompt_file = open(cfg.prompt_dir + cfg.data.dataset + "_" + cfg.prompt_type + ".txt", "r")
    system_template = prompt_file.read()
    human_template = "\n## Input \n> {post}"

    key_file = open(cfg.model.key_path, "r") 
    llm_key = key_file.read().rstrip('\n')
                
    llm = direct_gpt(model_name=cfg.model.llm_name, 
            openai_api_key=llm_key,
            system_template=system_template,
            human_template=human_template,
            temperature=cfg.model.temperature,
            max_tokens=None,
            seed=cfg.seed,
            response_format={"type": "json_object"}
            )
    
    return llm
        
@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    dataset = instantiate(cfg.data.dataset_class)
    llm = get_model(cfg)
    nest_asyncio.apply()
    llm_posts_df = pd.read_csv(cfg.data.posts_path, index_col=0).head(cfg.data.dataset_size)
    structured_columns, Y_cols, Z_cols = dataset.XYZ_names()
    structured_columns.extend([Y_cols, Z_cols, "post"])

    llm_samples_df = pd.DataFrame(columns = structured_columns)
    if "relevant" in cfg.data.posts_path:
        # hard filter out posts that do not contain treatment and outcome information
        llm_posts_df = dataset.hard_filter_ty(llm_posts_df)
    elif "inclusion" in cfg.prompt_type:
        # hard filter out posts that are known to not match inclusion criteria
        llm_posts_df = dataset.discretize(llm_posts_df)
        llm_posts_df = llm_posts_df.map(lambda x: np.nan if x in ["Unknown", "unknown"] else x)
    
    llm_inputs = []
    save_path = save_path_fn(cfg)

    if cfg.restore:
        llm_samples_df = pd.read_csv(save_path, index_col=0) 
        llm_posts_df = llm_posts_df.iloc[len(llm_samples_df):]

    for i, row in tqdm(llm_posts_df.iterrows()):
        post = row["post"]
        llm_inputs.append({"post": post})
        if len(llm_inputs) >= cfg.model.batch_size or len(llm_inputs) == cfg.data.dataset_size:
            llm_out_dicts = llm.get_outputs(llm_inputs)
            llm_out_dicts = [json.loads(text) for text in llm_out_dicts]
            dict_to_save = [{**llm_out_dicts[j], **{'post': llm_inputs[j]["post"]}} for j in range(len(llm_inputs))]
            df_to_save = pd.DataFrame.from_dict(dict_to_save)
            llm_samples_df = pd.concat([llm_samples_df, df_to_save], ignore_index=True)
            if "inclusion" in cfg.prompt_type:
                llm_posts_df.update(llm_samples_df, overwrite=False)
                llm_samples_df = llm_posts_df.copy().iloc[:len(llm_samples_df)]
                llm_samples_df = dataset.discretize(llm_samples_df, hard_filter=False, inf=False)
            llm_samples_df.to_csv(save_path) 
            llm_inputs = []


    save_path_final = save_path_fn(cfg, final=True)
    llm_samples_df.to_csv(save_path_final)

if __name__ == "__main__":
    main()