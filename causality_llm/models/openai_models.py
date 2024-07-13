from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
import asyncio
from openai import AsyncOpenAI

class gpt:
    def __init__(self,
                 model_name: str,
                 openai_api_key: str,
                 output_parser,
                 system_template: str='',
                 human_template: str='',
                 temperature: float=0.7,
                 top_p: float=1.0,
                 max_tokens: int=None,
                 n: int=1,
                 seed: int = 1234,
                 response_format=None):
        self.generator = ChatOpenAI(model_name=model_name,
                                    openai_api_key=openai_api_key,
                                    temperature=temperature,
                                    n=n,
                                    max_tokens=max_tokens,
                                    model_kwargs={'response_format': response_format,
                                                  'top_p': top_p, 
                                                  'seed': seed
                                                  },
                                    request_timeout=180
                                    )
        self.system_template = system_template
        self.human_template = human_template
        self.output_parser = output_parser
        self.chat_prompt = ChatPromptTemplate.from_messages([('system', system_template), 
                                                             ('human', human_template),])
        self.chat_chain = self.chat_prompt | self.generator | self.output_parser

    async def chat_chain_async_call(self, input_dicts):
        outputs = await self.chat_chain.abatch(input_dicts)
        return outputs

    def get_outputs(self, input_dicts):
        return asyncio.run(self.chat_chain_async_call(input_dicts))

class direct_gpt:
    def __init__(self,
                 model_name: str,
                 openai_api_key: str,
                 system_template: str='',
                 human_template: str='',
                 temperature: float=0.7,
                 top_p: float=1.0,
                 max_tokens: int=16,
                 seed: int = 1234,
                 response_format=None):
        
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model_name = model_name
        self.system_template = system_template
        self.human_template = human_template
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.response_format = response_format

    async def predict(self, user_input, extra_system=""):
        if self.human_template != "":
            user_input = self.human_template.format(**user_input)
        response = await self.client.chat.completions.create(
                                model = self.model_name,
                                messages=[
                                    { "role": "system",
                                    "content": self.system_template },
                                    { "role": "user",
                                    "content": user_input },
                                    { "role": "system",
                                    "content": extra_system } ],
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                                top_p=1,
                                response_format=self.response_format
        )
        return response.choices[0].message.content

    def get_outputs(self, input_dicts):
        return [asyncio.run(self.predict(input_dicts[k])) for k in range(len(input_dicts))]