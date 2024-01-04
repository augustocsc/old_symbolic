import torch
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "gpt2-10var"
DATABASE_NAME = "data.txt"
RUN_NAME = "4/1/24"
model_checkpoint="gpt2"



dataset = load_dataset("augustocsc/math-expressions", data_files=DATABASE_NAME)['train'].train_test_split(train_size=.9, test_size=.1)



tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                          bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>',
                                          pad_token='<|pad|>')
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

#block_size = tokenizer.model_max_length
block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

import multiprocessing
cores = multiprocessing.cpu_count()

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=300,
    num_proc=cores,
)

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
#this need bc we added tokens (bos_token, etc).
model.resize_token_embeddings(len(tokenizer))

from transformers import TrainingArguments, Trainer

torch.cuda.empty_cache()
training_args = TrainingArguments(num_train_epochs=5,
                            output_dir=MODEL_NAME,
                            resume_from_checkpoint=True,
                            run_name=RUN_NAME,
                            seed=42,
                            save_steps=1000,
                            save_strategy="epoch",
                            save_total_limit=2,
                            evaluation_strategy="steps",
                            eval_steps=200,
                            per_device_train_batch_size=64,
                            per_device_eval_batch_size=64,
                            auto_find_batch_size=True,
                            #optim= "adamw_apex_fused",
                            #learning_rate=5e-5,
                            #weight_decay=0.01,
                            push_to_hub=True,
                            hub_strategy = "checkpoint",
                            #max_steps=1000,
                            report_to="wandb",
                            overwrite_output_dir = True,)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets['train'],
    eval_dataset=lm_datasets['test']
)

trainer.train()
trainer.push_to_hub()
tokenizer.save_pretrained(MODEL_NAME, push_to_hub=True)