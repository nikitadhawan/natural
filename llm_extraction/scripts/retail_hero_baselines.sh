python baselines.py data=retail_hero \
                    data.split=val \
                    baseline_name=sentence_encoder \
                    data.posts_path=$HOME/natural/data/retail_hero/persona_posts_retail_hero_val_gpt-4-1106-preview_temp0.7_propensity0.7_drop0.1_persona_p0.5_posts.csv \
                    experiment_name=all_mpnet;

python baselines.py data=retail_hero \
                    data.split=val \
                    baseline_name=bow \
                    data.posts_path=$HOME/natural/data/retail_hero/persona_posts_retail_hero_val_gpt-4-1106-preview_temp0.7_propensity0.7_drop0.1_persona_p0.5_posts.csv \
                    experiment_name=count