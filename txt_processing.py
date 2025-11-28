import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def preprocess(txt_list):
    sw = set(stopwords.words('english'))
    p_txt = []
    for resume in txt_list:
        t = re.sub(r'\s+', ' ', resume or ' ')
        t = re.sub(r'[^a-zA-Z# ]', ' ', t)
        t = re.sub(r'[^\w\s]', ' ', t)
        toks = [w.lower() for w in t.split() if w.isalpha() and w.lower() not in sw]
        p_txt.append(" ".join(toks))
    return p_txt
