import random, sys
import numpy as np
from scipy.special import softmax
from llama import Llama, Dialog
from llama.tokenizer import Tokenizer
from llama3 import Llama as Llama3 
from llama3 import Dialog as Dialog3
from llama3.tokenizer import Tokenizer as Tokenizer3
from typing import List

class llama:
    def __init__(self,
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.8,
        top_p: float = 1.0,
        max_seq_len: int= 4096,
        max_gen_len: int = 512,
        max_batch_size: int = 128,
        random_seed: bool = True,
        system_template = '',
        seed: int = None):

        if random_seed:
            seed = random.randint(-sys.maxsize, sys.maxsize)
        elif seed:
            seed = seed
        else:
            seed = 1

        self.tokenizer = Tokenizer(tokenizer_path)
        self.system_template = system_template
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p
        self.max_batch_size = max_batch_size

        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            seed=seed)

    def tokenize(self, prompt, bos=False, eos=False):
        tokens_int = self.tokenizer.encode( prompt, bos, eos  )
        tokens = [ self.tokenizer.decode(token) for token in tokens_int  ]
        return tokens

    def add_system_template(self, prompts):
        results = []
        for prompt in prompts:
            results += [ self.system_template + ' \n' + prompt  ]
        return results

    def get_outputs(self, prompts, return_logprobs=False):
        prompts = self.add_system_template(prompts)
        completions = []
        logprobs = []
        results = self.generator.text_completion(
            prompts,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p, 
            logprobs=return_logprobs
        )
        for result in results:
            completions += [result['generation']]
            if return_logprobs: logprobs += [ sum(result['logprobs'])  ]
        if return_logprobs:
            return completions, logprobs
        return completions
    
    def compute_input_probs(self, X, options):
        logprobs = []
        X = self.add_system_template(X)   
        X_repeat = [x for x in X for _ in range(len(options))]
        options_repeat = options * len(X) 
        inp = []
        for x, y in zip(X_repeat, options_repeat):
            inp += [x+y]
            if len(inp) == self.max_batch_size or len(inp) == len(X_repeat):
                results = self.generator.text_completion(
                    inp,
                    max_gen_len=self.max_gen_len,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    logprobs=True,
                    echo=True
                )
                logprobs += [sum(result['logprobs']) for result in results]
                inp = []
        logprobs = np.array(logprobs).reshape((len(X), len(options)))
        probs = softmax(logprobs, axis=1)
        sample_idx = [np.random.choice(len(prob), p=prob) for prob in probs]
        max_idx = np.argmax(probs, axis=1)
        return probs, sample_idx, max_idx

    def compute_XY_logprobs(self, X, Y):
        prompts = []
        results = []
        X = self.add_system_template(X)    
        for x,y in zip(X,Y):
            prompts += [ x+y  ]
            if len(prompts) == self.max_batch_size or len(prompts) == len(X):
                results += self.generator.text_completion(
                    prompts,
                    max_gen_len=self.max_gen_len,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    logprobs=True,
                    echo=True
                )
                prompts = []
        logprobs = []
        for x,y,result in zip(X, Y, results):
            x_size = len(self.tokenize(x,bos=True))
            y_size = len(self.tokenize(y))
            y_logprobs = result['logprobs'][x_size-1:x_size+y_size-1]
            logprobs += [ sum(y_logprobs)  ]
        return logprobs

    def compute_XY_probs(self, X, options):
        X_repeat = [x for x in X for _ in range(len(options))]
        options_repeat = options * len(X)
        logprobs = self.compute_XY_logprobs(X_repeat, options_repeat)
        logprobs = np.array(logprobs).reshape((len(X), len(options)))
        probs =  softmax(logprobs, axis=1) 
        sample_idx = [np.random.choice(len(prob), p=prob) for prob in probs] 
        max_idx = np.argmax(probs, axis=1)
        return probs, sample_idx, max_idx


class chat_llama(llama):
    def __init__(self,
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.8,
        top_p: float = 1.0,
        max_seq_len: int= 4096,
        max_gen_len: int = 512,
        max_batch_size: int = 128,
        random_seed: bool = True,
        system_template = '',
        seed: int = None):

        if random_seed:
            seed = random.randint(-sys.maxsize, sys.maxsize)
        elif seed:
            seed = seed
        else:
            seed = 1

        self.tokenizer = Tokenizer(tokenizer_path)
        self.system_template = system_template
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p
        self.max_batch_size = max_batch_size

        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            seed=seed)

    def add_system_template(self, prompts):
        results: List[Dialog] = [ [ { 'role': 'system', 'content': self.system_template}, {'role': 'user', 'content': prompt } ]  for prompt in prompts ]
        return results

    def get_outputs(self, prompts, return_logprobs=False):
        prompts = self.add_system_template(prompts)
        completions = []
        logprobs = []
        results = self.generator.chat_completion(
            prompts,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p, 
            logprobs=return_logprobs
        )
        for result in results:
            if result['generation']['role'] == 'assistant':
                completions += [result['generation']['content']]
                if return_logprobs: logprobs += [ sum(result['logprobs']) ]
        if return_logprobs:
            return completions, logprobs
        else:
            return completions
        
    def compute_input_probs(self, X, options):
        logprobs= []
        X_repeat = [x for x in X for _ in range(len(options))]
        options_repeat = options * len(X) 
        inp = []
        for x, y in zip(X_repeat, options_repeat):
            inp += [x+y]
            if len(inp) == self.max_batch_size or len(inp) == len(X_repeat):
                inp = super().add_system_template(inp) 
                results = self.generator.text_completion(
                    inp,
                    max_gen_len=self.max_gen_len,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    logprobs=True,
                    echo=True
                )
                # ignore max_gen_len tokens
                logprobs += [sum(result['logprobs'][:-self.max_gen_len]) for result in results]
                inp = []
        logprobs = np.array(logprobs).reshape((len(X), len(options)))
        probs = softmax(logprobs, axis=1)
        sample_idx = [np.random.choice(len(prob), p=prob) for prob in probs]
        max_idx = np.argmax(probs, axis=1)
        return probs, sample_idx, max_idx
    

class chat_llama3(llama):
    def __init__(self,
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.8,
        top_p: float = 1.0,
        max_seq_len: int= 8000,
        max_gen_len: int = 512,
        max_batch_size: int = 128,
        random_seed: bool = True,
        system_template = '',
        seed: int = None):

        if random_seed:
            seed = random.randint(-sys.maxsize, sys.maxsize)
        elif seed:
            seed = seed
        else:
            seed = 1

        self.tokenizer = Tokenizer3(tokenizer_path)
        self.system_template = system_template
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p
        self.max_batch_size = max_batch_size

        self.generator = Llama3.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            seed=seed)

    def tokenize(self, prompt, bos=False, eos=False):
        tokens_int = self.tokenizer.encode( prompt, bos, eos  )
        tokens = [ self.tokenizer.decode(token) for token in tokens_int  ]
        return tokens
    
    def add_system_template(self, prompts):
        results: List[Dialog3] = [ [ { 'role': 'system', 'content': self.system_template}, {'role': 'user', 'content': prompt } ]  for prompt in prompts ]
        return results

    def get_outputs(self, prompts, return_logprobs=False):
        prompts = self.add_system_template(prompts)
        completions = []
        logprobs = []
        results = self.generator.chat_completion(
            prompts,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p, 
            logprobs=return_logprobs
        )
        for result in results:
            if result['generation']['role'] == 'assistant':
                completions += [result['generation']['content']]
                if return_logprobs: logprobs += [ sum(result['logprobs']) ]
        if return_logprobs:
            return completions, logprobs
        else:
            return completions
        
    def compute_input_probs(self, X, options):
        logprobs= []
        X_repeat = [x for x in X for _ in range(len(options))]
        options_repeat = options * len(X) 
        inp = []
        for x, y in zip(X_repeat, options_repeat):
            inp += [x+y]
            if len(inp) == self.max_batch_size or len(inp) == len(X_repeat):
                inp = super().add_system_template(inp) 
                results = self.generator.text_completion(
                    inp,
                    max_gen_len=self.max_gen_len,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    logprobs=True,
                    echo=True
                )
                # ignore max_gen_len tokens
                logprobs += [sum(result['logprobs'][:-self.max_gen_len]) for result in results]
                inp = []
        logprobs = np.array(logprobs).reshape((len(X), len(options)))
        probs = softmax(logprobs, axis=1)
        sample_idx = [np.random.choice(len(prob), p=prob) for prob in probs]
        max_idx = np.argmax(probs, axis=1)
        return probs, sample_idx, max_idx
    