from flask import Flask, render_template, request, redirect
import sumy
from sumy.summarizers.kl import KLSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer as Luhn
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
import numpy as np

app = Flask(__name__)
app.debug = True


def tf_idf_summarizer(text_corpus, num):
    nlp = English()
    nlp.add_pipe('sentencizer')
    doc = nlp(text_corpus.replace("\n", ""))
    sentences = [sent.text.strip() for sent in doc.sents]
    sentence_organizer = {k:v for v,k in enumerate(sentences)}
    tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                    strip_accents='unicode', 
                                    analyzer='word',
                                    token_pattern=r'\w{1,}',
                                    ngram_range=(1, 3), 
                                    use_idf=1,smooth_idf=1,
                                    sublinear_tf=1,
                                    stop_words = 'english')
    tf_idf_vectorizer.fit(sentences)
    sentence_vectors = tf_idf_vectorizer.transform(sentences)
    sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
    top_n_sentences = [sentences[int(ind)] for ind in np.argsort(sentence_scores, axis=0)[::-1][:int(num)]]
    mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
    mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
    ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
    summary = " ".join(ordered_scored_sentences)
    return summary

def luhn_summarizer(text, num):
    tokenizer = sumy.nlp.tokenizers.Tokenizer
    parser = sumy.parsers.plaintext.PlaintextParser.from_string(text, tokenizer('english'))

    summarizer = Luhn()
    summary = summarizer(parser.document, sentences_count = num)
    text = ''
    for sent in summary:
        text += str(sent)
    
    return text

def textrank_summarizer(text, num):
    parser = PlaintextParser.from_string(text,Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary =summarizer(parser.document,num)
    text_summary=""
    for sentence in summary:
        text_summary+=str(sentence)

    return text_summary

def klsum_summarizer(text, num):
    
    parser = PlaintextParser.from_string(text,Tokenizer("english"))
    summarizer_kl = KLSummarizer()
    summary =summarizer_kl(parser.document, num)
    kl_summary=""
    for sentence in summary:
        kl_summary+=str(sentence)  
    return kl_summary

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/form", methods=["POST"])
def form():
    text = request.form.get("input_text")
    method = request.form.get("methods")
    num = request.form.get("num")
    summary = ''
    if method == 'luhn':
        summary = luhn_summarizer(text, num)
        method = 'Luhn Summarizer'
    if method == 'textrank':
        summary = textrank_summarizer(text, num)
        method = 'TextRank Summarizer'
    if method == 'klsum':
        summary = klsum_summarizer(text, num)
        method = 'KL-Sum Summarizer'
    if method == 'tfidf':
        summary = tf_idf_summarizer(text, num)
        method = 'TF-IDF based Summarizer'

    return render_template('summary.html', summary = summary, method = method)
    