import nltk 
import time 
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from math import log10 
from nltk.corpus import stopwords
from xml.dom.minidom import parse
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict, Counter

# Download supporting data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define variables for tokenizing, stemming and removing stop words
ss = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'\w+')
stoplist = stopwords.words('english')

def processing_words(source_data):
    tokens = tokenizer.tokenize(source_data.lower())
    terms = []

    for token in tokens:
        if not token.isalpha():
            continue
        if token in stoplist:
            continue
        terms.append(ss.stem(token))

    return terms

def making_inverted_index(corpus):
    index = defaultdict(Counter)

    for i in range(len(corpus)):
        for term in corpus[i]:
            index[term][i + 1] += 1

    return index

def DF(term, index):
    return len(index[term])

def IDF(corpus, index):
    N = len(corpus)
    idf = {}
    for term in index.keys():
        dft = DF(term, index)
        idf[term] = log10(1 + ((N - dft + 0.5) / (dft + 0.5)))

    return idf

def RSV_weighting(corpus, index, med, idf=None, k1=None, b=None):
    if med != 'bim1' and med != 'bim2' and med != 'bm25':
        return

    w = {}

    if med == 'bim1' or med == 'bim2':
        N = len(corpus)
        for term in index.keys():
            dft = DF(term, index)
            if med == 'bim1':
                pt = 0.5
            elif med == 'bim2':
                pt = 1 / 3 + 2 / 3 * dft / N
            w[term] = log10(pt / (1 - pt)) + log10(N / dft)
    else:
        sum_Ld = 0
        for doc in corpus:
            sum_Ld += len(doc)
        Lave = sum_Ld / len(corpus)

        for doc_id, doc in enumerate(corpus):
            Ld = len(doc)
            w[doc_id] = {}

            for term in doc:
                w[doc_id][term] = (index[term][doc_id + 1] * idf[term] * (k1 + 1)) / (
                        index[term][doc_id + 1] + k1 * ((1 - b) + b * (Ld / Lave)))

    return w

class Dataset:
    def __init__(self, path, tag):
        self.collection = parse(path).getElementsByTagName(tag)

    def getContent(self, id):
        try:
            return self.collection[id].firstChild.nodeValue
        except:
            return ''

    def getLength(self):
        return len(self.collection)
    

class PIR:
    def __init__(self, corpus, med, k1=None, b=None):
        self.original_corpus = [corpus.getContent(id).split()
                                for id in range(corpus.getLength())]
        self.corpus = [processing_words(corpus.getContent(id))
                       for id in range(corpus.getLength())]

        self.index = making_inverted_index(self.corpus)
        self.methods = med

        if self.methods == 'bm25':
            self.k1 = k1
            self.b = b
            self.idf = IDF(self.corpus, self.index)
            self.weight = RSV_weighting(self.corpus, self.index, 
                                        med, self.idf, self.k1, self.b)
        else:
            self.weight = RSV_weighting(self.corpus, self.index, med)

        self.ranked = []
        self.query_text = ''
        self.N_retrieved = 0

    def RSV_document_query(self, doc_id, query):
        RSV = 0
        doc = self.corpus[doc_id]

        for term in doc:
            if term in query:
                if self.methods == 'bm25':
                    RSV += self.weight[doc_id][term]
                else:
                    RSV += self.weight[term]

        return RSV

    def ranking(self, query):
        docs = set()
        for term in query:
            if term in self.index:
                docs = docs.union(set(self.index[term].keys()))

        scores = []
        for doc in docs:
            scores.append((doc, self.RSV_document_query(doc - 1, query)))

        self.ranked = sorted(scores, key=lambda x: x[1], reverse=True)

        return self.ranked

    def recompute_weight(self, rel_idx, query):
        rel_docs = []
        for idx in rel_idx:
            doc_id = self.ranked[idx - 1][0] - 1
            rel_docs.append(doc_id)

        N = len(self.corpus)
        V = len(rel_docs)

        for term in query:
            if term in self.index.keys():
                Vt = 0

                d = []
                for doc_id in rel_docs:
                    if term in self.corpus[doc_id]:
                        Vt += 1
                        if self.methods == 'bm25':
                            d.append(doc_id)

                pt = (Vt + 0.5) / (V + 1)
                ut = (DF(term, self.index) - Vt + 0.5) / (N - V + 1)

                if self.methods == 'bm25':
                    sum_Ld = 0
                    for doc in self.corpus:
                        sum_Ld += len(doc)
                    Lave = sum_Ld / len(self.corpus)

                    for doc_id in d:
                        Ld = len(self.corpus[doc_id])
                        ct = log10(pt / (1 - pt)) + log10((1 - ut) / ut)
                        self.weight[doc_id][term] = self.index[term][doc_id + 1] * ct * (self.k1 + 1) / (
                                self.index[term][doc_id + 1] + self.k1 * ((1 - self.b) + self.b * (Ld / Lave)))
                else:
                    self.weight[term] = log10(pt / (1 - pt)) + log10((1 - ut) / ut)

    def answer_query(self, query_text, disp=False):
        self.query_text = query_text
        query = processing_words(query_text)
        ranking = self.ranking(query)

        self.N_retrieved = 15

        links = Dataset(r'D:\web\links_fix.xml', tag='link')
        res = pd.DataFrame(columns = ['Rank', 'DocID', 'Content', 'Score'])

        # Print Retrieved Documents
        if disp:
            n_retr = self.N_retrieved if self.N_retrieved <= len(self.ranked) else len(self.ranked)
            if n_retr == 0:
                self.N_retrieved = 0
                print('No retrieved paper.')
            else:
                for i in range(n_retr):
                    doc, score = ranking[i]

                    text = self.original_corpus[doc - 1]
                    if len(text) > 10:
                        text = text[:10]
                        text.append('...')
                    text = ' '.join(text)

                    link = 'https://openaccess.thecvf.com/' + links.getContent(doc-1)
                    

                    #st.write(f"{(i + 1):4d}. Document {doc:4d}, score = {score:5.3f}")
                    # st.write('=' * (len(text) + 6))
                    st.markdown(f"<h6 style='color: #151e3d;'>{i+1}. {text}</h6>", unsafe_allow_html=True)
                    st.markdown(link, unsafe_allow_html=True)
                    #st.write(links.getContent(doc))
                    #st.write('\n')

                    #res.loc[len(res)] = [i+1, doc, text, score]
                #return res
                #st.markdown(res.style.set_table_styles([dict(selector='*', props=[('text-align', 'center')]), dict(selector='th', props=[('min-width', '100px')])]).to_html(),unsafe_allow_html=True)

        if self.methods == 'bm25':
            self.weight = RSV_weighting(self.corpus, self.index, self.methods, self.idf, self.k1, self.b)
        else:
            self.weight = RSV_weighting(self.corpus, self.index, self.methods)

    def relevance_feedback_and_reanswer(self, rel_fb, disp=False):
        if self.query_text == '':
            print('Cannot get feedback before a query is formulated.')
            return

        query = processing_words(self.query_text)
        self.recompute_weight(rel_fb, query)
        self.answer_query(self.query_text, disp)

    def read_document(self, rank_num):
        if (self.query_text == ''):
            print('Cannot select a document before a query is formulated.')
            return

        text = self.original_corpus[self.ranked[rank_num - 1][0] - 1]
        pos = 0
        lmax = 0

        print(f"Rank {rank_num} is document {self.ranked[rank_num - 1][0]}, score: {self.ranked[rank_num - 1][1]:5.3f}")
        while pos <= len(text):
            t = text[pos:pos + 15]
            t = ' '.join(t)
            lmax = max(lmax, len(t))
            pos += 15

        pos = 0
        print("=" * (lmax + 6))
        while pos <= len(text):
            t = text[pos:pos + 15]
            t = ' '.join(t)
            print(f"|| {t}" + " " * (lmax - len(t)) + " ||")
            pos += 15
        print("=" * (lmax + 6))

    def show_more(self):
        res = pd.DataFrame(columns = ['Rank', 'DocID', 'Content', 'Score'])

        if (self.N_retrieved + 10 > len(self.ranked)):
            st.write('No more documents available.')
            return

        for i in range(self.N_retrieved, self.N_retrieved + 10):
            doc, score = self.ranked[i]

            text = self.original_corpus[doc - 1]
            if len(text) > 15:
                text = text[:15]
                text.append('...')
            text = ' '.join(text)

            #st.write(f"Rank {(i + 1):4d}: Document {doc:4d}, score = {score:5.3f}")
            #st.write('=' * (len(text) + 6))
            st.write(f"{text}")
            #st.write('=' * (len(text) + 6) + '\n')
            #res.loc[len(res)] = [i+1, doc, text, score]
        #st.markdown(res.style.set_table_styles([dict(selector='*', props=[('text-align', 'center')]), dict(selector='th', props=[('min-width', '100px')])]).to_html(),unsafe_allow_html=True)

        self.N_retrieved += 10

# Read example dataset
titles = Dataset(r'D:\web\papers.xml', tag='doc')

# Create model with pt = 0.5
ex_bim_model = PIR(titles, med='bim1')

st.markdown("<h1 style='text-align: center; color: #255FDB;'>CSearch</h1>", unsafe_allow_html=True)
st.write("<h3 style='text-align: center; color: #3f3f3f;'> &#127759 Search For The Future &#127759</h3>", unsafe_allow_html=True)

form = st.form(key='my_form')
user_input = form.text_input(label='Looking for something?')
submit_button = form.form_submit_button(label='Search')

if submit_button:
    ex_bim_model.answer_query(user_input, disp=True)


