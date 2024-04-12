from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

# GPT-2 does not have a pad token by default, so we use eos_token as pad_token.
# This is necessary for models that don't have a PAD token.
tokenizer.pad_token = tokenizer.eos_token

# Define the text you want to continue or generate text based on
input_text = "Once upon a time in a faraway land,"

# Tokenize the input text. Now with proper padding handling.
inputs = tokenizer.encode_plus(
    input_text, 
    return_tensors="tf", 
    add_special_tokens=True, 
    padding='max_length', # Use 'max_length' for padding
    max_length=512 # Specify the max_length for models that require a specific input size
)

# Generate text with explicit pad_token_id
outputs = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=200,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id # Setting pad_token_id to eos_token_id for open-end generation
)

# Decode the generated text and print it
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)