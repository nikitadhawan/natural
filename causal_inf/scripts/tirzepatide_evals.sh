# NATURAL IPW
python evaluate.py data=tirzepatide \
                    data.probs_data_path$HOME/natural/data/tirzepatide/tirzepatide_llama-70b-chat_prob_hybrid_ipw_probs.csv \
                    data.sample_data_path=$HOME/natural//data/tirzepatide/sample_inclusion_tirzepatide_unknowns_gpt-4-turbo-preview_samples.csv \
                    estimator@toeval=ipw_hybrid \
                    data.dataset_size=1000 \
                    'data.eval_size=[1000]' \
                    experiment_name=gpt4_llama2_70b;

# NATURAL OI
python evaluate.py data=tirzepatide \
                    data.probs_data_path$HOME/natural/data/tirzepatide/tirzepatide_llama-70b-chat_prob_hybrid_oi_probs.csv \
                    data.sample_data_path=$HOME/natural//data/tirzepatide/sample_inclusion_tirzepatide_unknowns_gpt-4-turbo-preview_samples.csv \
                    estimator@toeval=oi_hybrid \
                    data.dataset_size=1000 \
                    'data.eval_size=[1000]' \
                    experiment_name=gpt4_llama2_70b;

# NATURAL MC IPW
python evaluate.py data=tirzepatide \
                    data.sample_data_path=$HOME/natural//data/tirzepatide/sample_inclusion_tirzepatide_unknowns_gpt-4-turbo-preview_samples.csv \
                    estimator@toeval=sample_ipw \
                    data.dataset_size=1000 \
                    'data.eval_size=[1000]' \
                    experiment_name=gpt4;

# NATURAL MC OI
python evaluate.py data=tirzepatide \
                    data.sample_data_path=$HOME/natural//data/tirzepatide/sample_inclusion_tirzepatide_unknowns_gpt-4-turbo-preview_samples.csv \
                    estimator@toeval=sample_standardization \
                    data.dataset_size=1000 \
                    'data.eval_size=[1000]' \
                    experiment_name=gpt4   