from Expression import Expression
from Experiment import Experiment
import os

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

def save_epoch(epoch, rewards, exprs, r2):
    index = [f'{line["name"]}_{epoch}_{i}' for i in range(len(rewards))]
    log_saved = {'index':index, 'expr':exprs, 'reward': rewards, 'r2':r2}
    df = pd.DataFrame(log_saved)
    df.to_csv('log_saved.csv', index=False, header=False)

def save_results(results, line, save_dir):
    df = pd.DataFrame(results)
    df.columns = line.keys()
    #if the file results.csv exists, append the results, otherwise create the file
    if os.path.exists(f'{save_dir}/results.csv'):
        df.to_csv(f'{save_dir}/results.csv', mode='a', index=False, header=False)
    else:
        df.to_csv(f'{save_dir}/results.csv', index=False)

   
class Agent:
    def __init__(self, params):

        self.config = PPOConfig(
            model_name="augustocsc/gpt-base", #add gpt2 model name here
            learning_rate=params.lr, #experiment with different lr?
            #log_with=None, #"wandb",
            mini_batch_size = params.mini_batch_size, # incase of memory issues while sampling, reduce batch size
            batch_size=params.batch_size,
            seed = params.seed,
            
        )

        self.save_dir = params.save_dir
        self.experiment_file = params.experiment_file


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #, torch_dtype=torch.float16
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.config.model_name)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.epochs = params.epochs

        self.stop_condition_type = params.stop_condition
        self.curr_epoch = 0
        self.max_reward = 0

        #self.model.bfloat16()
        #self.ref_model.bfloat16()

    def reward_pipeline(self, response_tensors, data):
        prefixes = [self.tokenizer.decode(response, skip_special_tokens=True) for response in response_tensors]
        exprs = [Expression(prefix.strip().split(" "), data) for prefix in prefixes]

        rewards = [torch.tensor(float(expr.score)) for expr in exprs]
        r2 = [expr.r2 for expr in exprs]

        return exprs, rewards, r2

    def stop_condition(self):
        if self.stop_condition_type == "epoch":                 
            print(f'Current epoch: {self.curr_epoch}')         
            return not self.curr_epoch >= self.epochs 
        #implement other stop conditions

    def train(self):

        prompts = ['Y' for i in range(self.config.batch_size)]
        encoded_prompts = self.tokenizer.batch_encode_plus(prompts, return_tensors='pt')

        #read the file data.csv
        data = pd.read_csv(self.experiment_file)
        results = []

        #for each line in data pandas dataframe
        for _, line in data.iterrows():    

            #define adam lr scheduler
            
            ppo_trainer = PPOTrainer(self.config, self.model, self.ref_model, self.tokenizer)


            self.device = ppo_trainer.accelerator.device
            if ppo_trainer.accelerator.num_processes == 1:
                self.device = 0 if torch.cuda.is_available() else "cpu" # to avoid a `pipeline` bug

            experiment = Experiment(experiment=line)
            print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("Working with expression: ", experiment.name)
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            #create a folder with the name of the expression in the save_dir folder
            if not os.path.exists(f'{self.save_dir}/{experiment.name}'):
                print(f'Creating folder {experiment.name}')
                os.mkdir(f'{self.save_dir}/{experiment.name}')

            log_saved = mean = top = []
            log_saved = {'index':[], 'expr':[], 'reward': [], 'r2':[]}
            max_reward = mean_reward = max_r2 = 0 
            output_min_length = 4
            output_max_length = 20
            output_length_sampler = LengthSampler(output_min_length, output_max_length)
            
            generation_kwargs = {
                "min_length":-1,
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            itt = 0
            start_time = time.time()
            end_time = 0

            
            self.curr_epoch = 0
            while self.stop_condition():
                max_expr = ""
                
                print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print("Working with expression: ", experiment.expression)
                print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                print(f'\nmax mean: {mean_reward}\n max top: {max_reward}\n')

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

                    response = ppo_trainer.generate(query.to(self.device), **generation_kwargs)
                    response_tensors.append(response.squeeze()[-gen_len:])
                
                batch['response'] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]
                
                # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
                exprs, rewards, r2 = self.reward_pipeline(response_tensors, experiment)
                
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
                with open(f'{self.save_dir}/{experiment.name}/log.txt', 'a') as f:
                    f.write(f'mean: {np.mean(rewards)}\ntop: {current_max_reward}\ninvalid: {rewards.count(0)}\n\n')
                

                #check if the max reward is greater than the previous max reward
                if current_max_reward > max_reward:
                    max_reward = current_max_reward
                    max_expr = current_max_expr         #TODO max expr is not being saved
                    max_r2 = current_max_r2
                
                #check if the mean reward is greater than the previous mean reward
                if np.mean(rewards) > mean_reward:
                    mean_reward = np.mean(rewards)

                #save the rewards and expressions
                index = [f'{line["name"]}_{self.curr_epoch}_{i}' for i in range(len(rewards))]
                
                log_saved['index'].extend(index)
                log_saved['expr'].extend(exprs)
                log_saved['reward'].extend(rewards)
                log_saved['r2'].extend(r2)
                
                #save log_saved in a csv file
                df = pd.DataFrame(log_saved)
                df.to_csv(f'{self.save_dir}/{experiment.name}/log_saved.csv', index=False, header=False)
                self.curr_epoch += 1


            end_time = time.time()

            line["nrmse"] = max_reward
            line["result"] = max_expr
            line["mean_nrmse"] = np.mean(rewards)
            line["r2"] = max_r2
            line["epoch"] = self.curr_epoch
            line["time"] = end_time - start_time
            
            results.append(line.to_list())

            print("Saving results...")
            #append the results in a csv file
            save_results(results, line, save_dir=self.save_dir)
            print("Results saved!")

            
            


