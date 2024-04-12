from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the pre-trained T5 model and tokenizer
model_name = "t5-small"  # Consider using 't5-base' or 't5-large' for potentially better performance
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define your input text with a clear task for the model
input_text = "translate English to Chinese: Once upon a time in a faraway land, a small village was known for its magical fruit trees. One day,"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)

# Adjust max_length if your input text is long to ensure there's room for generated content
desired_output_length = 150  # This includes the length of your input text
inputs_length = inputs.input_ids.size(1)
max_length = inputs_length + desired_output_length

# Generate text with optimized parameters for more creative output
outputs = model.generate(
    inputs["input_ids"], 
    max_length=max_length, 
    do_sample=True, 
    temperature=0.9, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=1
)

# Decode the generated text and print it
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)