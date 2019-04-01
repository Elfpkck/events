#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re

import numpy as np
import spacy
from spacy import displacy
from spacy.attrs import ORTH, LEMMA, NORM, TAG
from spacy.matcher import Matcher
from tabulate import tabulate

nlp = spacy.load('en_core_web_md')


# In[2]:


def create_custom_tokenizer(nlp):
    custom_prefixes = [
#         r'[a-zA-Z]+(?=\d)', 
        r'\d+(?=[a-zA-Z])'
    ]
    all_prefixes_re = spacy.util.compile_prefix_regex(list(nlp.Defaults.prefixes) + custom_prefixes)

    custom_infixes = []
    infix_re = spacy.util.compile_infix_regex(list(nlp.Defaults.infixes) + custom_infixes)
    
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)   
    
    return spacy.tokenizer.Tokenizer(nlp.vocab, 
                     nlp.Defaults.tokenizer_exceptions,
                     prefix_search = all_prefixes_re.search, 
                     infix_finditer = infix_re.finditer, suffix_search = suffix_re.search,
                     token_match=None)


# In[3]:


nlp.tokenizer = create_custom_tokenizer(nlp)


# In[4]:


text = """43yo term P0010 induction for cholestasis of pregnancy, gestational
hypertension, on cytotec; category 1 tracing, 120baseline, accels, no decels,
moderate variability, toco 4/10. GBS positive on ampicillin; Rh negative s/p
Rhogam 11/2018, EFW 2551g 55%ile 12/19/18 sono, dated by LMP c/w 8w4d sono.
Significant medical/ surgical history for scoliosis s/p rods placement 1997.
I accepted care from XXXXXXXXXX"""
doc = nlp(text)


# In[5]:


for sent in doc.sents:
#     print(sent)
    for tk in sent:
        print(tk)


# In[6]:


sent_i = 0
sents = list(doc.sents)
displacy.render(sents[sent_i], style='dep', jupyter=True, options={"distance": 100})


# In[7]:


get_ipython().system(f'jupyter nbconvert --to script custom_tokenizer')


# In[ ]:




