from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Define your input text with a mask token
input_text = "Once upon a time in a [MASK] fertile land, people live happily."

# Tokenize the input text and generate attention masks
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Predict the masked token
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
predictions = outputs.logits

# Find the predicted token
predicted_index = torch.argmax(predictions[0, input_ids[0] == tokenizer.mask_token_id], dim=1)
predicted_token = tokenizer.decode(predicted_index)

# Replace the mask token with the model's prediction
generated_text = input_text.replace("[MASK]", predicted_token)

print(generated_text)