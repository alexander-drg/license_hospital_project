import re
import nltk
from nltk.corpus import stopwords

# download once if missing (safe to keep)
def _ensure_nltk():
    pkgs = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for res, name in pkgs:
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(name)

_ensure_nltk()

def preprocess(txt):
    """
    Returns a preprocessed list of texts.
    """
    sw = set(stopwords.words("english"))
    space_pattern = r"\s+"
    special_letters = r"[^a-zA-Z#]"
    p_txt = []

    for resume in txt:
        text = re.sub(space_pattern, " ", resume)            # remove extra spaces
        text = re.sub(special_letters, " ", text)            # remove special chars
        text = re.sub(r"[^\w\s]", "", text)                  # remove punctuation
        text = text.split()
        text = [w for w in text if w.isalpha()]              # keep alphabetic
        text = [w for w in text if w.lower() not in sw]      # remove stopwords
        text = [w.lower() for w in text]                     # lowercase
        p_txt.append(" ".join(text))

    return p_txt
