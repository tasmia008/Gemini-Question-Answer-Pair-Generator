# Python 3.10.14
import torch
import pandas as pd
from trl import SFTTrainer
from unsloth import FastLanguageModel
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("/home/bikas/qa-gimini/Llama3/file.csv")
# dataset = dataset.iloc[:10000]
# Split the data into training, validation, and test sets
train, X_temp = train_test_split(dataset, test_size=0.3, random_state=42)
validation, test = train_test_split(X_temp, test_size=0.10, random_state=42)

train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)
test = test.reset_index(drop=True)

# Create a datasets.Dataset object
train = Dataset.from_dict(train)
validation = Dataset.from_dict(validation)
test = Dataset.from_dict(test)

# Create DatasetDict
dataset = DatasetDict({
    'validation': validation,
    'test': test,
    'train': train
})


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.float16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/llama-3-8b-bnb-4bit",
    model_name = "meta-llama/Meta-Llama-3-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    token = "",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


alpaca_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(example):
    # Retrieve question and answer from the example
    instruction = "Please provide a detailed answer to the following question"
    question = example["Question"]
    answer = example["Answer"]

    # Check the structure and content of the example
    # print(f"Question: {question}")
    # print(f"Answer: {answer}")

    # Construct the formatted prompt text
    prompt_text = alpaca_prompt.format(instruction, question, answer) + EOS_TOKEN

    # Return the formatted prompt text as a dictionary
    return {"text": prompt_text}

# Assuming 'dataset' is your dataset object
dataset = dataset.map(formatting_prompts_func)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        num_train_epochs = 2,
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = True,
        bf16 = False,
        logging_steps = 100,
        evaluation_strategy='epoch',
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()
model.save_pretrained("/home/bikas/qa-gimini/Llama3/Bangla-Text-Question-Answer") # Local saving