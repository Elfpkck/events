#!/usr/bin/env python
# coding: utf-8

# Plan:
# * NLP Tasks
# * Popular Techniques
# * Word Embedding

# ## NLP Tasks:
# * Information extraction:
#     * Named entity recognition (NER)
#     * Coreference resolution
#     * Relationship extraction
# * Information retrieval
# * Sentiment analysis
# * Natural language generation
# * Machine translation
# * Recommender System
# * Question answering
# * Text Classification

# # Popular Techniques

# ## Pipeline Architecture for an Information Extraction System
# https://www.nltk.org/book/ch07.html

# ![ie-architecture](https://www.nltk.org/images/ie-architecture.png)

# ## Sentence Segmentation

# In[1]:


import nltk


# In[2]:


s = """Joining / merging on duplicate keys can cause a returned frame that is the multiplication of the row 
dimensions, which may result in memory overflow. It is the user’ s responsibility to manage duplicate values 
in keys before joining large DataFrames."""


# In[3]:


nltk.sent_tokenize(s)


# ### Numbered list problem

# In[4]:


s = """1. Joining / merging on duplicate keys can cause a returned frame that is the multiplication of the row 
dimensions, which may result in memory overflow. 2. It is the user’ s responsibility to manage duplicate values 
in keys before joining large DataFrames."""


# In[5]:


nltk.sent_tokenize(s)


# ### Regular expressions and solving of numbered list problem

# In[6]:


r"^(?:(?:(?:0?[13578]|1[02])(\/|-|\.)31)\1|(?:(?:0?[13-9]|1[0-2])(\/|-|\.)(?:29|30)\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:0?2(\/|-|\.)29\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:(?:0?[1-9])|(?:1[0-2]))(\/|-|\.)(?:0?[1-9]|1\d|2[0-8])\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$"


# ![alt text](img\wtf.jpg "WTF")

# In[7]:


import re


# In[8]:


bool(re.match(r"\d+\.", "4."))


# In[9]:


# why we have to escape dot character?
bool(re.match(r"\d+.", "4b"))


# https://regex101.com/r/F8dY80/3

# #### End of string

# In[10]:


sent_text = nltk.sent_tokenize(s)
# sent_text.append("4. New sentence")

for sent in sent_text:
    if re.match(r"\d+\.", sent):
        print(sent)


# In[11]:


def _separate_by_sent(text: str):
    sent_text = nltk.sent_tokenize(text)

    for i, sent in enumerate(sent_text):
        sent_text[i] = re.sub(
            r"\d+\.$",
            lambda x: '' if len(x.group()) < 5 else x.group(),
            sent)

    return [x for x in sent_text if x]


# In[12]:


sentences = _separate_by_sent(s)
sentences


# ## Tokenization

# In[13]:


tk_sent = [nltk.word_tokenize(sent) for sent in sentences]
tk_sent


# ## Part of Speech (POS) Tagging

# In[14]:


tk_pos_sent = [nltk.pos_tag(sent) for sent in tk_sent]
tk_pos_sent[0]


# ### Get POS by sentences not by tokens!

# In[15]:


tk_pos_sent[0] == [nltk.pos_tag([tk]) for tk in tk_sent[0]]


# ## Lemmatization

# In[16]:


from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()

for sent in tk_sent:
    for tk in sent:
        tk = tk.lower()
        lemma = lmtzr.lemmatize(tk)  # without POS
#         print(f"{tk} ---> {lemma}")
        if tk != lemma:
            print(f"{tk} ---> {lemma}")


# In[17]:


def _wnpos(pos: str) -> str:
    """Transform nltk POS to wordnet POS."""
    pos = pos.lower()
    wnpos = "n"

    if pos.startswith("j"):
        wnpos = "a"
    elif pos[0] in ('n', 'r', 'v'):
        wnpos = pos[0]

    return wnpos


# In[18]:


for sent in tk_pos_sent:
    for tk, pos in sent:
        print(pos, _wnpos(pos))
#         tk = tk.lower()
#         lemma = lmtzr.lemmatize(tk, _wnpos(pos))
        
#         if tk != lemma:
#             print(f"{tk} ---> {lemma}")


# ## Stemming

# In[19]:


sno = nltk.stem.SnowballStemmer('english')

for sent in tk_sent:
    for tk in sent:
        tk = tk.lower()
        stem = sno.stem(tk)
        print(f"{tk} ---> {stem}")


# ## Stopwords

# In[20]:


from nltk.corpus import stopwords

set(stopwords.words('english'))


# ## n-grams

# In[21]:


for sent in tk_sent:
    n_grams = nltk.ngrams(sent, 2)
    
    for grams in n_grams:
        print(grams)


# ## Punctuation

# In[22]:


from string import punctuation
punctuation


# ## SpaCy

# ![alt text](img\spacy.png "spacy")

# In[23]:


import spacy

# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en')
doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
# 
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)


# ![alt text](img\spacy2.png "spacy2")

# In[24]:


from spacy import displacy

nlp = spacy.load('en_core_web_sm')
text = """But Google is starting from behind. The company made a late push
into hardware, and Apple’s Siri, available on iPhones, and Amazon’s Alexa
software, which runs on its Echo and Dot devices, have clear leads in
consumer adoption."""
doc = nlp(text)
displacy.render(doc, style='ent', jupyter=True, options={"distance": 120})


# In[25]:


doc = nlp("This is a sentence")
# doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
displacy.render(doc, style='dep', jupyter=True, options={"distance": 120})


# ## Google NLP

# https://cloud.google.com/natural-language/

# # Word Embedding
# * bag of words
# * TF-IDF
# * hash embeddings
# * word2vec, FastText, GloVe

# ## Indexing

# In[26]:


corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]


# In[27]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()   
vect = vectorizer.fit_transform(corpus)


# In[ ]:


sorted(vectorizer.vocabulary_.items())


# ## Bag of Words / TF-IDF

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(use_idf=True)  # switcher
bow = transformer.fit_transform(vect)


# In[ ]:


from scipy.sparse import csr_matrix
import pandas as pd

inverted = {v: k for k, v in vectorizer.vocabulary_.items()}
df = pd.DataFrame(csr_matrix.todense(bow))
df.columns = [inverted[col] for col in df.columns]
df


# In[ ]:


df.max()


# In[ ]:


df.max()


# In[ ]:


get_ipython().system(u'jupyter nbconvert --to script nlp_presentation.ipynb')


# In[ ]:




