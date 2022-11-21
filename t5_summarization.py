import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

def t5_summarizer(raw_docx):
    raw_text = raw_docx
    # Instantiating the model and tokenizer 
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')

    preprocess_text = raw_text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt", max_length=512, truncation=True).to(device)

	# summmarize 
    summary_ids = model.generate(tokenized_text,
										num_beams=4,
										no_repeat_ngram_size=2,
										min_length=40,
										max_length=150,
										early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary