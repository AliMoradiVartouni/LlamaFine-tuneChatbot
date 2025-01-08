import torch
print(torch.version.cuda)
print(torch.backends.cudnn.enabled)
print(torch.cuda.is_available(), torch.__version__)
print(torch.cuda.get_device_name(0))
# assert torch.cuda.get_device_capability()[0] >= 8
print(torch.cuda.get_device_capability())
print(torch.backends.cudnn.is_available())  # Should return True
print(torch.backends.cudnn.version())      # Should return the cuDNN version

import os
# os.environ["CUDA_HOME"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
# os.environ["PATH"] = os.environ["PATH"] + ";C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin"

from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format, SFTTrainer
from peft import LoraConfig
from peft import AutoPeftModelForCausalLM

from huggingface_hub import login
# from datasets import load_dataset
from datasets import Dataset

assert torch.cuda.get_device_capability()[0] >= 8, 'Hardware not supported for Flash Attention'
# install flash-attn


token_path = r"/home/ali/moradi/token_huggingface.txt"
with open(token_path, encoding='utf8') as file_object:
    access_token = file_object.read()
login(
  token=access_token, # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)

# Convert dataset to OAI messages
system_message = """You are a chatbot. Users will ask you questions in English and you will response based on the any title of Conference information.
Title:
{title}"""
def create_conversation(sample):
  return {
    "messages": [
      {"role": "system", "content": system_message.format(schema=sample["context"])},
      {"role": "user", "content": sample["question"]},
      {"role": "assistant", "content": sample["answer"]}
    ]
  }


file_path = r"/home/ali/moradi/Conference_Q&A.txt"
import json

# Open and read the JSON file
try:
    # Read the .txt file
    with open(file_path, "r", encoding="utf-8") as txt_file:
        data = txt_file.read()  # Read the text content
    dataset=json.loads(data)

except json.JSONDecodeError as e:
    print("Error: The content of the .txt file is not valid JSON.")
    print(f"Details: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")



dataset = Dataset.from_dict({"messages": [item["conversation"] for item in dataset]})
# Print the data
print(dataset)
print(dataset[0])  # Example: View the first record
print(dataset[345]["messages"])

# save datasets to disk
dataset.to_json("train_dataset.json", orient="records")
# dataset["test"].to_json("test_dataset.json", orient="records")
#######################################################################################


model_id = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "Llama-3.2-3B"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    # llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
)


# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # model_path,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    # max_memory=max_memory,
    quantization_config=bnb_config
)
print(torch.cuda.memory_allocated())

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right' # to prevent warnings

# # set chat template to OAI chatML, remove if you start from a fine-tuned model
model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)


newModel_path = "/home/ali/moradi/models/Radman-Llama-3.2-3B"
args = TrainingArguments(
    output_dir=newModel_path, # directory to save and repository id
    num_train_epochs=10,                     # number of training epochs
    per_device_train_batch_size=2,          # batch size per device during training
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_hf",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=1e-5,                     # learning rate, based on QLoRA paper
    # fp16=False,  # Disable mixed precision
    bf16=True,                              # use bfloat16 precision
    # tf32=True,                              # use tf32 precision
    max_grad_norm=1.0,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.1,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",           # use constant learning rate scheduler
    weight_decay=0.1,                    # Add weight decay
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)


max_seq_length = 2048 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

model.resize_token_embeddings(len(tokenizer))
# save model
# trainer.save_model()
trainer.save_model()  # Save locally

###################################################### save ###########################################
# free the memory again
del model
del trainer
torch.cuda.empty_cache()

extra_path = os.path.join(newModel_path, "extra")
# Ensure the directories exist
os.makedirs(extra_path, exist_ok=True)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(newModel_path)

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
model.resize_token_embeddings(len(tokenizer))
# Save merged model and tokenizer to the `extra` folder
merged_model.save_pretrained(extra_path, safe_serialization=True, max_shard_size="2GB")
tokenizer.save_pretrained(extra_path)
print(f"Model and tokenizer successfully saved in: {extra_path}")




