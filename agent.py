import Expression

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
tqdm.pandas()

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from transformers import  AutoTokenizer

class Agent:
    def __init__(self, model_name="augustocsc/gpt-base", learning_rate=1.41e-5, log_with="wandb", mini_batch_size = 8, tokenizer="gpt2"):
        self.tokenizer = tokenizer
        self.config = PPOConfig(
                            model_name=model_name,
                            learning_rate=learning_rate,
                            log_with=log_with,
                            mini_batch_size = mini_batch_size
                        )
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.config.model_name)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        prompts = ['Y' for i in range(self.config.batch_size)]
        self.encoded_prompts = tokenizer.batch_encode_plus(prompts, return_tensors='pt')
        self.ppo_trainer = PPOTrainer(self.config, self.model, self.ref_model, self.tokenizer)

    def reward_pipeline(self, response_tensors, data):

        prefixes = [self.tokenizer.decode(response, skip_special_tokens=True) for response in response_tensors]
        exprs = [Expression(prefix.strip().split(" "), data) for prefix in prefixes]

        rewards = [torch.tensor(float(expr.score)) for expr in exprs]

        return exprs, rewards




device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
   device = 0 if torch.cuda.is_available() else "cpu" # to avoid a `pipeline` bug

rewards_saved = []
expr_saved = []
mean = []
top = []
output_min_length = 4
output_max_length = 10
output_length_sampler = LengthSampler(output_min_length, output_max_length)
epochs = 50

generation_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}


for epoch in tqdm(range(epochs)):
    query_tensors = encoded_prompts['input_ids']
    query_tensors = list(torch.tensor(query_tensors))

    batch = dict()
    #### Get response from gpt2
    response_tensors = []

    batch['query'] = query_tensors


    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query.to(device), **generation_kwargs)

        response_tensors.append(response.squeeze()[-gen_len:])
    batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    exprs, rewards = reward_pipeline(response_tensors, data)

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

