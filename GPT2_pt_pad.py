from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define your input text
input_text = "Once upon a time in a faraway land,"

# Tokenize the input text and generate attention masks
inputs = tokenizer(input_text, return_tensors="pt")
attention_mask = inputs["attention_mask"]

# Generate text
outputs = model.generate(inputs["input_ids"], attention_mask=attention_mask, max_length=20, do_sample=True, top_k=50, top_p=0.95)

# Decode the generated text and print it
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)