python llm_sample.py \
        data=liraglutide \
        data.posts_path=$HOME/natural/data/liraglutide/filter_posts_liraglutide_gpt-3.5-turbo_relevance_filtered_posts_relevant.csv \
        model=gpt \
        model.llm_name=gpt-3.5-turbo-0125 \
        prompt_type=sample_tofilter \
        experiment_name=filter_ty;

python llm_sample.py \
        data=liraglutide \
        data.posts_path=$HOME/natural/data/liraglutide/sample_tofilter_liraglutide_filter_ty_gpt-3.5-turbo-0125_samples.csv \
        model=gpt \
        model.batch_size=1 \
        model.llm_name=gpt-4-turbo-preview \
        prompt_type=sample_incontext \
        experiment_name=knowns;

python llm_sample.py \
        data=liraglutide \
        data.posts_path=$HOME/natural/data/liraglutide/sample_incontext_liraglutide_knowns_gpt-4-turbo-preview_samples.csv \
        model=gpt \
        model.llm_name=gpt-4-turbo-preview \
        model.temperature=1 \
        model.batch_size=1 \
        prompt_type=sample_inclusion \
        experiment_name=unknowns;


torchrun --nproc_per_node 8 llm_conditional.py \
                                data=liraglutide \
                                model=llama \
                                model.llm_name=llama-70b-chat \
                                use_samples=ipw \
                                prompt_type=prob_hybrid \
                                data.posts_path=$HOME/natural/data/liraglutide/sample_inclusion_liraglutide_unknowns_gpt-4-turbo-preview_samples.csv  \
                                data.samples_path=$HOME/natural/data/liraglutide/sample_inclusion_liraglutide_unknowns_gpt-4-turbo-preview_samples.csv \
                                experiment_name=ipw;

torchrun --nproc_per_node 8 llm_conditional.py \
                                data=liraglutide \
                                model=llama \
                                model.llm_name=llama-70b-chat \
                                use_samples=oi \
                                prompt_type=prob_hybrid \
                                data.posts_path=$HOME/natural/liraglutide/sample_inclusion_liraglutide_unknowns_gpt-4-turbo-preview_samples.csv  \
                                data.samples_path=$HOME/natural/data/liraglutide/sample_inclusion_unknowns_liraglutide_gpt-4-turbo-preview_samples.csv \
                                experiment_name=oi;
