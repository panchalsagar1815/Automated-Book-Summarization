from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def pegasus_summarization(raw_docx):
    raw_text = raw_docx

    # Load tokenizer 
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")

    # Load model 
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")

    # Create tokens - number representation of our text
    tokens = tokenizer(raw_text, truncation=True, padding="longest", return_tensors="pt")

    # Summarize 
    summary = model.generate(**tokens)

    summary_text = tokenizer.decode(summary[0])

    return summary_text