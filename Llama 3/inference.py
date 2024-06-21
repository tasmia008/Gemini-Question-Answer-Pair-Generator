from unsloth import FastLanguageModel
import torch
alpaca_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Bikas0/Bengali-Question-Answer-Llama3", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Please provide a detailed answer to the following question", # instruction
        "বাংলা একাডেমি আইন কোন কারণে সদস্যপদ বাতিল করা হবে ?", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 2048)





# import torch
# from transformers import TextStreamer
# from unsloth import FastLanguageModel

# alpaca_prompt = """
# ### Instruction:
# {}

# ### Input:
# {}

# ### Response:
# {}"""

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="Bikas0/Bengali-Question-Answer-Llama3", # YOUR MODEL YOU USED FOR TRAINING
#     max_seq_length=2048,
#     dtype=torch.float16,
#     load_in_4bit=True,
# )

# FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# inputs = tokenizer(
#     [
#         alpaca_prompt.format(
#             "Please provide a detailed answer to the following question", # instruction
#             "জমাক্রত অর্থ কোন কোন ক্ষেত্রে ব্যবহার করা যায়?", # input
#             "", # output - leave this blank for generation!
#         )
#     ], 
#     return_tensors="pt"
# ).to("cuda")

# text_streamer = TextStreamer(tokenizer)
# generated_response = model.generate(**inputs, streamer=text_streamer, max_new_tokens=2048)

# decoded_response = tokenizer.decode(generated_response[0], skip_special_tokens=True)
# decoded_response = decoded_response.split("### Response:")[-1]
# print(decoded_response)
