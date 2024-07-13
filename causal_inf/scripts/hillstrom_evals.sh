# NATURAL Full
python evaluate.py data=hillstrom \
                    data.split=val \
                    data.probs_data_path=$HOME/natural/data/hillstrom/hillstrom_llama-70b-chat_prob_full_val_probs.csv \
                    estimator@toeval=ipw_enum \
                    data.dataset_size=2000 \
                    'data.eval_size=[2000]' \
                    experiment_name=llama2_70b_val;

# NATURAL IPW
python evaluate.py data=hillstrom \
                    data.split=val \
                    data.sample_data_path=$HOME/natural/data/hillstrom/sample_incontext_hillstrom_val_gpt-4-turbo-preview_samples.csv \
                    data.probs_data_path=$HOME/natural/data/hillstrom/hillstrom_llama-70b-chat_prob_hybrid_val_ipw_probs.csv \
                    estimator@toeval=ipw_hybrid \
                    data.dataset_size=2000 \
                    'data.eval_size=[2000]' \
                    experiment_name=gpt4_llama2_70b_val;

# NATURAL OI
python evaluate.py data=hillstrom \
                    data.split=val \
                    data.sample_data_path=$HOME/natural/data/hillstrom/sample_incontext_hillstrom_val_gpt-4-turbo-preview_samples.csv \
                    data.probs_data_path=$HOME/natural/data/hillstrom/hillstrom_llama-70b-chat_prob_hybrid_val_oi_probs.csv \
                    estimator@toeval=oi_hybrid \
                    data.dataset_size=2000 \
                    'data.eval_size=[2000]' \
                    experiment_name=gpt4_llama2_70b_val;

# NATURAL MC IPW
python evaluate.py data=hillstrom \
                    data.sample_data_path=$HOME/natural/data/hillstrom/sample_incontext_hillstrom_val_gpt-4-turbo-preview_samples.csv \
                    data.split=val \
                    estimator@toeval=sample_ipw \
                    data.dataset_size=2000 \
                    'data.eval_size=[2000]' \
                    experiment_name=gpt4_val; 

# NATURAL MC OI
python evaluate.py data=hillstrom \
                    data.sample_data_path=$HOME/natural/data/hillstrom/sample_incontext_hillstrom_val_gpt-4-turbo-preview_samples.csv \
                    data.split=val \
                    estimator@toeval=sample_standardization \
                    data.dataset_size=2000 \
                    'data.eval_size=[2000]' \
                    experiment_name=gpt4_val; 

# Baselines
python evaluate.py data=hillstrom \
                    data.split=val \
                    data.sample_data_path=$HOME/natural/data/hillstrom/hillstrom_val_all_mpnet_sentence_encoder_preds.csv \
                    estimator@toeval=sample_ipw \
                    baseline_name=sentence_encoder \
                    data.dataset_size=2000 \
                    'data.eval_size=[2000]' \
                    experiment_name=allmpnet_val;  

python evaluate.py data=hillstrom \
                    data.split=val \
                    data.sample_data_path=$HOME/natural/data/hillstrom/hillstrom_val_count_bow_preds.csv \
                    estimator@toeval=sample_ipw \
                    baseline_name=bow \
                    data.dataset_size=2000 \
                    'data.eval_size=[2000]' \
                    experiment_name=count_val 