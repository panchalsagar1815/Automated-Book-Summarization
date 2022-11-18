from __future__ import unicode_literals
from flask import Flask,render_template,url_for,request
from spacy_summarization import text_summarizer
# from gensim.summarization import summarize
# from nltk_summarization import nltk_summarizer
from t5_summarization import t5_summarizer
from bert_summarization import bert_summarizer
from pegasus_large import pegasus_summarization
import PyPDF2
import os
import time
import spacy
nlp = spacy.load('en_core_web_sm')
app = Flask(__name__)

# Web Scraping Pkg
from bs4 import BeautifulSoup
# from urllib.request import urlopen
from urllib.request import urlopen

# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Sumy 
def sumy_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


# Reading Time
def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/analyze',methods=['GET','POST'])
def analyze():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

app.config['PDF_UPLOADS'] = 'E:\\FOM\\Sem 3\\Applied Project 2\\NLP-Web-Apps-master\\Summaryzer_Text_Summarization_App\\static\\uploads'
app.config['ALLOWED_DOC_EXTENSION'] = ["pdf"]

@app.route('/analyze_file',methods=['GET','POST'])
def analyze_file():
	start = time.time()
	if request.method == 'POST':
		name = ''
		if request.files:
			doc = request.files["pdf"]
			doc.save(os.path.join(app.config["PDF_UPLOADS"], doc.filename))

		sample_pdf = open(r''+os.path.join(app.config["PDF_UPLOADS"], doc.filename), mode='rb')
		
		pdfdoc = PyPDF2.PdfFileReader(sample_pdf)

		for i in range(pdfdoc.numPages):
			current_page = pdfdoc.getPage(i)
			# print("===================")
			# print("Content on page:" + str(i + 1))
			# print("===================")
			# print(current_page.extractText())
			rawtext = current_page.extractText()
		
		final_summary = t5_summarizer(rawtext)
		# raw_url = request.form['raw_url']
		# rawtext = get_text(raw_url)
		final_reading_time = readingTime(rawtext)
		# final_summary = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)



@app.route('/compare_summary')
def compare_summary():
	return render_template('compare_summary.html')

@app.route('/comparer',methods=['GET','POST'])
def comparer():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary_spacy = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary_spacy)
		# Gensim Summarizer
		# final_summary_gensim = summarize(rawtext)
		# summary_reading_time_gensim = readingTime(final_summary_gensim)
		# T5 Transformers
		final_summary_t5 = t5_summarizer(rawtext)
		summary_reading_time_t5 = readingTime(final_summary_t5)
		# BERT
		final_summary_bert = bert_summarizer(rawtext)
		summary_reading_time_bert = readingTime(final_summary_bert)

		# Pegasus
		final_summary_pegasus = pegasus_summarization(rawtext)
		summary_reading_time_pegasus = readingTime(final_summary_pegasus)
		# NLTK
		# final_summary_nltk = nltk_summarizer(rawtext)
		
		# Sumy
		# final_summary_sumy = sumy_summary(rawtext)
		 

		end = time.time()
		final_time = end-start
	return render_template('compare_summary.html',ctext=rawtext,final_summary_spacy=final_summary_spacy,final_summary_bert=final_summary_bert,final_summary_t5=final_summary_t5,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time,summary_reading_time_bert=summary_reading_time_bert,final_summary_pegasus=final_summary_pegasus,summary_reading_time_pegasus=summary_reading_time_pegasus,summary_reading_time_t5=summary_reading_time_t5)



@app.route('/about')
def about():
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)