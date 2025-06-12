import re
import spacy
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS

# ─────────────────────────────────────────────────────────────────────────────
# 0) Ensure required NLTK data is present
# ─────────────────────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load spaCy model
# ─────────────────────────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ─────────────────────────────────────────────────────────────────────────────
# 2) Define the two source texts
# ─────────────────────────────────────────────────────────────────────────────
text1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""

text2 = """During our final discuss, I told him about the new submission — the one we were waiting since
last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit πριν
he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome με strong coffee and future
targets"""

# ─────────────────────────────────────────────────────────────────────────────
# 3) Pipeline definitions using 3 different libraries
# ─────────────────────────────────────────────────────────────────────────────

# 3.1 spaCy: tokenize, remove stopwords, lemmatize
def pipeline_spacy(text: str) -> list[str]:
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]

# 3.2 NLTK: RegexpTokenizer + stopwords + WordNet lemmatization
lemmatizer = WordNetLemmatizer()
nltk_stop = set(stopwords.words("english"))
regexp_tokenizer = RegexpTokenizer(r"[A-Za-z]+")

def pipeline_nltk(text: str) -> list[str]:
    tokens = regexp_tokenizer.tokenize(text)
    return [
        lemmatizer.lemmatize(tok.lower())
        for tok in tokens
        if tok.lower() not in nltk_stop
    ]

# 3.3 Gensim: simple_preprocess (lowercase, deacc), remove Gensim stopwords
def pipeline_gensim(text: str) -> list[str]:
    return [
        tok
        for tok in simple_preprocess(text, deacc=True)
        if tok not in GENSIM_STOPWORDS
    ]

# ─────────────────────────────────────────────────────────────────────────────
# 4) Collect and apply pipelines
# ─────────────────────────────────────────────────────────────────────────────
pipelines = {
    "spaCy Lemmatization":       pipeline_spacy,
    "NLTK RegexpTokenizer":      pipeline_nltk,
    "Gensim simple_preprocess":  pipeline_gensim,
}

def reconstruct(text: str, pipeline_fn) -> str:
    tokens = pipeline_fn(text)
    return " ".join(tokens)

if __name__ == "__main__":
    for name, fn in pipelines.items():
        recon1 = reconstruct(text1, fn)
        recon2 = reconstruct(text2, fn)

        print(f"\n=== {name} Reconstruction ===\n")
        print("Text 1:\n", recon1, "\n")
        print("Text 2:\n", recon2, "\n")
