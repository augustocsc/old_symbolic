from Expression import Expression
from Experiment import Experiment
import re

import time
import torch
from tqdm import tqdm
import pandas as pd
tqdm.pandas()

import pandas as pd
import numpy as np
from numpy import sqrt, exp, cos, sin, tan, log, pi, e

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler, respond_to_batch


config = PPOConfig(
    model_name="augustocsc/gpt-base", #add gpt2 model name here
    learning_rate=1.41e-5, #experiment with different lr?
    #log_with=None, #"wandb",
    mini_batch_size = 16, # incase of memory issues while sampling, reduce batch size
    batch_size=256,
)

def reward_pipeline(response_tensors, data):
    prefixes = [tokenizer.decode(response, skip_special_tokens=True) for response in response_tensors]
    exprs = [Expression(prefix.strip().split(" "), data) for prefix in prefixes]

    rewards = [torch.tensor(float(expr.score)) for expr in exprs]
    r2 = [expr.r2 for expr in exprs]

    return exprs, rewards, r2

#all loaded from the pre-trained model
#AutoModelForCausalLMWithValueHead is cus
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
#def avaliation():

prompts = ['Y' for i in range(config.batch_size)]
encoded_prompts = tokenizer.batch_encode_plus(prompts, return_tensors='pt')

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)


device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu" # to avoid a `pipeline` bug

#read the file data.csv
data = pd.read_csv('data/test_expr.csv')
results = []

#for each line in data pandas dataframe
for _, line in data.iterrows():    
    experiment = Experiment(experiment=line)
    print("Working with expression: ", experiment.expression)

    log_saved = mean = top = []
    log_saved = {'index':[], 'expr':[], 'reward': [], 'r2':[]}
    max_reward = mean_reward = 0 
    output_min_length = 4
    output_max_length = 50
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    
    generation_kwargs = {
        "min_length":-1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    itt = 0
    start_time = time.time()
    end_time = 0

    #epochs = 10
    #for epoch in tqdm(range(epochs)):
    epoch = 0
    while max_reward < 0.9 or mean_reward < 0.5:
        max_expr = ""
        max_reward = max_r2 = 0

        #query_tensors = tokenizer(encoded_prompts['input_text'], padding=True, truncation=True, return_tensors="pt").input_ids
        query_tensors = encoded_prompts['input_ids']
        query_tensors = list(query_tensors.clone().detach())
        
        batch = dict()
        #### Get response from gpt2
        response_tensors = []

        batch['query'] = query_tensors

        for query in tqdm(query_tensors):
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len

            response = ppo_trainer.generate(query.to(device), **generation_kwargs, batch_size=8)
            response_tensors.append(response.squeeze()[-gen_len:])
        
        batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        exprs, rewards, r2 = reward_pipeline(response_tensors, experiment)
        
        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        ppo_trainer.log_stats(stats, batch, rewards)

        #sort the rewards and expressions
        dict_exprs = dict(zip(rewards, exprs))
        dict_exprs = sorted(dict_exprs, reverse=True)
    
        #storing the max reward and expression
        max_index = np.argmax(rewards)
        current_max_reward = rewards[max_index]
        current_max_expr = exprs[max_index]
        current_max_r2 = r2[max_index]

        print(f'mean: {np.mean(rewards)}\ntop: {current_max_reward}\n')
        #print how many None expressions
        print(f'invalid: {rewards.count(0)}\n')

        #save mean, top and invalid into a txt file
        with open('log.txt', 'a') as f:
            f.write(f'mean: {np.mean(rewards)}\ntop: {current_max_reward}\ninvalid: {rewards.count(0)}\n\n')
        

        #check if the max reward is greater than the previous max reward
        if current_max_reward > max_reward:
            max_reward = current_max_reward
            max_expr = current_max_expr
            max_r2 = current_max_r2

        #save the rewards and expressions
        index = [f'{line["name"]}_{epoch}_{i}' for i in range(len(rewards))]
        
        log_saved['index'].extend(index)
        log_saved['expr'].extend(exprs)
        log_saved['reward'].extend(rewards)
        log_saved['r2'].extend(r2)

        epoch += 1


    end_time = time.time()

    line["nrmse"] = max_reward
    line["result"] = max_expr
    line["mean_nrmse"] = np.mean(rewards)
    line["r2"] = max_r2
    line["epoch"] = epoch
    line["time"] = end_time - start_time
    
    results.append(line.to_list())

    print("Saving results...")
    #save the results in a csv file
    df = pd.DataFrame(results)
    df.columns = line.keys()
    df.to_csv('results.csv', index=False, header=False)

    #save log_saved in a csv file
    df = pd.DataFrame(log_saved)
    df.to_csv('log_saved.csv', index=False, header=False)
