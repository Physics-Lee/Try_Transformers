from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

# Define the text you want to continue or generate text based on
input_text = "Once upon a time in a faraway land,"

# Tokenize the input text
inputs = tokenizer.encode(input_text, return_tensors="tf")

# Generate text
outputs = model.generate(inputs, max_length=20, do_sample=True, top_k=50, top_p=0.95)

# Decode the generated text and print it
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)