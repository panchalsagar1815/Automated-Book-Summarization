from summarizer import Summarizer,TransformerSummarizer

def bert_summarizer(raw_docx):
    raw_text = raw_docx

    preprocess_text = raw_text.strip().replace("\n","")
    bert_model = Summarizer()
    bert_summary = ''.join(bert_model(preprocess_text, min_length=30))
    # print(bert_summary)
    return bert_summary