# NATURAL

</div>

This repository is an implementation of NATURAL in [End-To-End Causal Effect Estimation from Unstructured Natural Language Data](https://arxiv.org/abs/2407.07018).

______________________________________________________________________

## Set-up

```bash
git clone https://github.com/nikitadhawan/natural.git
cd natural
conda create --name natural_env --file requirements.txt
conda activate natural_env
python setup.py install
```

### Open AI models
To use Open AI models, request a key to use the API and copy it into `natural/key.txt`.

### LLAMA models
Follow instructions in the [LLAMA2](https://github.com/meta-llama/llama) and [LLAMA3](https://github.com/meta-llama/llama3) repositories to download model weights.

______________________________________________________________________

## Generate synthetic posts or filter real data

See `post_generation/scripts`.

The resulting datasets are also available for download in `data/`.

## Extract samples and/or conditionals using LLMs

See `llm_extraction/scripts`.

## Evaluate ATE and other metrics

See `causal_inf/scripts`.
