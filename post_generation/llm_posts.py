import os
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

from langchain.schema import BaseOutputParser
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from causality_llm.models import gpt
from causality_llm.utils import shuffle_dict, drop_feat

import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', FutureWarning)


def save_path_fn(cfg, final=False):
    save_path = cfg.save_path + cfg.data.dataset + "/"
    if final:
        save_path += cfg.date_time + "_"
    save_path += cfg.prompt_type + "_" + cfg.data.dataset + "_" + cfg.data.split + "_" + cfg.model.llm_name + '_' 
    save_path += 'temp' + str(cfg.model.temperature) + '_propensity' + str(cfg.data.prop_score) 
    save_path += '_drop' + str(cfg.data.drop_prob) + '_persona_p' + str(cfg.persona_p) + "_" + cfg.experiment_name 
    save_path += '_posts.csv'
    return save_path

class OutputParser(BaseOutputParser):
    """Parse the output of an LLM call to natural language."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text

def get_model(cfg):
    prompt_file = open(cfg.prompt_dir + cfg.data.dataset + "_" + cfg.prompt_type + ".txt", "r")
    system_template = prompt_file.read()
    human_template = "\n## Attributes \nThe following are attributes that you have, along with their descriptions. \n> {features} \n"
    if "persona" in cfg.prompt_type:
        human_template += "\n## Personality Traits \nThe following dictionary describes your personality with levels (High or Low) of the Big Five personality traits. \n> {traits} \n"
    human_template += "\n## Your Instructions \nWrite a social media post in first-person, accurately describing the information provided. Write this post in the tone and style of someone with the given personality traits, without simply listing them. \nOnly return the post that you can broadcast on social media and nothing more. \n\n## Post \n>"

    key_file = open(cfg.model.key_path, "r") 
    openai_key = key_file.read().rstrip('\n')
    llm = gpt(model_name=cfg.model.llm_name, 
              openai_api_key=openai_key,
              output_parser=OutputParser(),
              system_template=system_template,
              human_template=human_template,
              temperature=cfg.model.temperature,
              n=cfg.model.sample_size, 
              seed=cfg.seed)
    return llm

@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    dataset = instantiate(cfg.data.dataset_class)
    X, Z, Y = dataset.get_data_split(transform=False, obs=cfg.data.observational)[cfg.data.split]
    llm = get_model(cfg)

    X_cols, Y_cols, Z_cols = dataset.XYZ_names()
    column_names = X_cols + [Y_cols, Z_cols, "llm_post"]
    trait_names = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
    if "persona" in cfg.prompt_type: 
        column_names += trait_names
    llm_posts_df = pd.DataFrame(columns=column_names)
    input_dicts, llm_input_dicts = [], []

    save_path = save_path_fn(cfg)
    start_index = 0

    if cfg.restore:
        llm_posts_df = pd.read_csv(save_path)[column_names]
        start_index = len(llm_posts_df)

    for i in tqdm(range(start_index, len(X))):
        all_features = X.iloc[[i]].dropna(axis="columns")
        all_features = all_features.to_dict('records')[0]
        treatment = Z.iloc[[i]].values[0]
        outcome = Y.iloc[[i]].values[0]
        all_features = drop_feat(all_features, cfg.data.drop_prob, X_cols)
        if treatment:
            all_features[Z_cols] = treatment
        all_features[Y_cols] = outcome
        all_features = shuffle_dict(all_features)
        features_text = dataset.post_gen_specs(all_features)

        if "persona" in cfg.prompt_type:
            levels = ["Low" if np.random.choice(2, p=[1-cfg.persona_p, cfg.persona_p]) == 0 else "High" for _ in range(5)]
            traits_dict = {"Extraversion": levels[0], 
                           "Agreeableness": levels[1], 
                           "Conscientiousness": levels[2], 
                           "Neuroticism": levels[3], 
                           "Openness": levels[4]}
            traits_text = str(traits_dict)
            llm_input_dicts.append({"features": features_text, "traits": traits_text})
            all_features.update(traits_dict)
        else:
            llm_input_dicts.append({"features": features_text})
        input_dicts.append(all_features)

        if len(llm_input_dicts) >= cfg.model.batch_size or len(llm_input_dicts) == cfg.data.dataset_size:
            llm_posts = llm.get_outputs(llm_input_dicts)
            dict_to_save = [{**input_dicts[j], **{'llm_post': llm_posts[j]}} for j in range(len(llm_input_dicts))]
            df_to_save = pd.DataFrame.from_dict(dict_to_save)
            llm_posts_df = pd.concat([llm_posts_df, df_to_save], ignore_index=True)
            llm_posts_df.to_csv(save_path) 
            input_dicts, llm_input_dicts = [], []
        

    save_path_final = save_path_fn(cfg, final=True)
    llm_posts_df.to_csv(save_path_final) 

if __name__ == "__main__":
    main()