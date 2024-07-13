python baselines.py data=hillstrom \
                    data.split=val \
                    baseline_name=sentence_encoder \
                    data.posts_path=$HOME/natural/data/hillstrom/persona_posts_hillstrom_val_gpt-4-1106-preview_temp0.7_propensity0.3_drop0.1_persona_p0.5_posts.csv \
                    experiment_name=all_mpnet;

python baselines.py data=hillstrom \
                    data.split=val \
                    baseline_name=bow \
                    data.posts_path=$HOME/natural/data/hillstrom/persona_posts_hillstrom_val_gpt-4-1106-preview_temp0.7_propensity0.3_drop0.1_persona_p0.5_posts.csv \
                    experiment_name=count