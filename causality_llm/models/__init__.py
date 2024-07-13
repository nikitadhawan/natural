from causality_llm.models.causal_models import *
from causality_llm.models.llama_models import *
from causality_llm.models.openai_models import *
from causality_llm.models.hf_models import *
from causality_llm.models.guided_models import *
from causality_llm.models.claude_models import *

CAUSAL_MODELS = {
    "naive": DifferenceInMeans,
    "ipw": IPSW,
    "standardization": OutcomeImputation,
    "strat_std": StratifiedOutcomeImputation,
    "dr_vanilla": DoublyRobust,
    "dr_ipfeature": DoublyRobustIPFeature,
    "dr_importance": DoublyRobustImportance
}

LLAMA_CKPT = {
    "llama-7b": "~/natural/weights/llama2_models/llama-2-7b",
    "llama-13b": "~/natural/weights/llama2_models/llama-2-13b", 
    "llama-70b": "~/natural/weights/llama2_models/llama-2-70b", 
    "llama-7b-chat": "~/natural/weights/llama2_models/llama-2-7b-chat",
    "llama-13b-chat": "~/natural/weights/llama2_models/llama-2-13b-chat", 
    "llama-70b-chat": "~/natural/weights/llama2_models/llama-2-70b-chat", 
    "llama2_tokenizer": "~/natural/weights/llama2_models/tokenizer.model",
    "llama3-8b-chat": "~/natural/weights/llama3_models/Meta-Llama-3-8B-Instruct", 
    "llama3-70b-chat": "~/natural/weights/llama3_models/Meta-Llama-3-70B-Instruct", 
    "llama3-8b-chat_tokenizer": "~/natural/weights/llama3_models/Meta-Llama-3-8B-Instruct/tokenizer.model", 
    "llama3-70b-chat_tokenizer": "~/natural/weights/llama3_models/Meta-Llama-3-70B-Instruct/tokenizer.model",
}
