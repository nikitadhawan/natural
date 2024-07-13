# NATURAL FULL
torchrun --nproc_per_node 8 llm_population.py \
                                data=retail_hero \
                                data.posts_path=$HOME/natural/data/retail_hero/persona_posts_retail_hero_val_gpt-4-1106-preview_temp0.7_propensity0.3_drop0.1_persona_p0.5_posts.csv \
                                model=llama \
                                enum_x=True \
                                prompt_type=prob_full \
                                experiment_name=val;

# NATURAL Sample-only / Hybrid
python llm_sample.py \
        data=retail_hero \
        data.posts_path=$HOME/natural/data/retail_hero/persona_posts_retail_hero_val_gpt-4-1106-preview_temp0.7_propensity0.3_drop0.1_persona_p0.5_posts.csv \
        model=gpt \
        model.llm_name=gpt-4-turbo-preview \
        prompt_type=sample_incontext \
        experiment_name=val;

torchrun --nproc_per_node 8 llm_conditional.py \
                                data=retail_hero \
                                model=llama \
                                model.llm_name=llama-70b-chat \
                                use_samples=ipw \
                                prompt_type=prob_hybrid \
                                data.posts_path=$HOME/natural/data/retail_hero/persona_posts_retail_hero_val_gpt-4-1106-preview_temp0.7_propensity0.3_drop0.1_persona_p0.5_posts.csv \
                                data.samples_path=$HOME/natural/data/retail_hero/sample_incontext_retail_hero_val_gpt-4-turbo-preview_samples.csv \
                                experiment_name=val_ipw;

torchrun --nproc_per_node 8 llm_conditional.py \
                                data=retail_hero \
                                model=llama \
                                model.llm_name=llama-70b-chat \
                                use_samples=oi \
                                prompt_type=prob_hybrid \
                                data.posts_path=$HOME/natural/data/retail_hero/persona_posts_retail_hero_val_gpt-4-1106-preview_temp0.7_propensity0.3_drop0.1_persona_p0.5_posts.csv \
                                data.samples_path=$HOME/natural/data/retail_hero/sample_incontext_retail_hero_val_gpt-4-turbo-preview_samples.csv \
                                experiment_name=val_oi;