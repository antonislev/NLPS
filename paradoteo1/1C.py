import re
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS

# ─────────────────────────────────────────────────────────────────────────────
# 0) Download any required NLTK data
# ─────────────────────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load spaCy model
# ─────────────────────────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

# ─────────────────────────────────────────────────────────────────────────────
# 2) Original texts
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
# 3) Pipeline definitions
# ─────────────────────────────────────────────────────────────────────────────

# 3.1 Simple regex tokenizer (original baseline)
def pipeline_regex_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text.lower())

# 3.2 spaCy: tokenize, remove stopwords, lemmatize
def pipeline_spacy(text: str) -> list[str]:
    doc = nlp(text)
    return [
        tok.lemma_.lower()
        for tok in doc
        if tok.is_alpha and not tok.is_stop
    ]

# 3.3 NLTK: RegexpTokenizer + stopwords + WordNet lemmatizer
lemmatizer = WordNetLemmatizer()
nltk_stop  = set(stopwords.words("english"))
regexp_tok = RegexpTokenizer(r"[A-Za-z]+")

def pipeline_nltk(text: str) -> list[str]:
    toks = regexp_tok.tokenize(text)
    return [
        lemmatizer.lemmatize(tok.lower())
        for tok in toks
        if tok.lower() not in nltk_stop
    ]

# 3.4 Gensim: simple_preprocess + Gensim stopwords
def pipeline_gensim(text: str) -> list[str]:
    return [
        tok
        for tok in simple_preprocess(text, deacc=True)
        if tok not in GENSIM_STOPWORDS
    ]

pipelines = {
    "Regex":    pipeline_regex_words,
    "spaCy":    pipeline_spacy,
    "NLTK":     pipeline_nltk,
    "Gensim":   pipeline_gensim,
}

# ─────────────────────────────────────────────────────────────────────────────
# 4) Compute metrics
# ─────────────────────────────────────────────────────────────────────────────
# Original token sets for Jaccard
orig1 = set(pipeline_regex_words(text1))
orig2 = set(pipeline_regex_words(text2))

records = []
for name, fn in pipelines.items():
    for label, text, orig in [("Text1", text1, orig1), ("Text2", text2, orig2)]:
        toks   = fn(text)
        total  = len(toks)
        unique = len(set(toks))
        ttr    = unique / total if total else 0.0
        jacc   = len(set(toks) & orig) / len(orig) if orig else 0.0

        records.append({
            "Pipeline":     name,
            "Text":         label,
            "TotalTokens":  total,
            "UniqueTokens": unique,
            "TTR":          ttr,
            "Jaccard":      jacc
        })

df = pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Display table and plots
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Comparison Table ===")
print(
    df
    .set_index(["Pipeline","Text"])
    .sort_index()
    .round({"TTR":3, "Jaccard":3})
)

for metric in ["TTR","Jaccard"]:
    plt.figure(figsize=(6,4))
    for label, grp in df.groupby("Text"):
        plt.plot(grp["Pipeline"], grp[metric], marker="o", label=label)
    plt.title(f"{metric} by Pipeline")
    plt.xlabel("Pipeline")
    plt.ylabel(metric)
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()

plt.show()
