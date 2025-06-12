import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import gensim.downloader as api
import torch
from transformers import BertTokenizer, BertModel
import spacy
from nltk.corpus import wordnet as wn
from scipy.spatial.distance import cosine

# ─── ορισμός αρχικών κειμένων ─────────────────────────────────────────────────
text1_orig = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""
text2_orig = """During our final discuss, I told him about the new submission — the one we were waiting since
last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit πριν
he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome με strong coffee and future
targets"""

# ─── ορισμός reconstructions dict ─────────────────────────────────────────────
reconstructions = {
    "Whitespace Split":   {"Text1": "today is our dragon boat festival, in our chinese culture, to celebrate it with all safe and great in our lives. hope you too, to enjoy it as my deepest wishes. thank your message to show our words to the doctor, as his next contract checking, to all of us. i got this message to see the approved message. in fact, i have received the message from the professor, to show me, this, a couple of days ago. i am very appreciated the full support of the professor, for our springer proceedings publication", "Text2": "during our final discuss, i told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? anyway, i believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. we should be grateful, i mean all of us, for the acceptance and efforts until the springer link came finally last week, i think. also, kindly remind me please, if the doctor still plan for the acknowledgments section edit πριν he sending again. because i didn’t see that part final yet, or maybe i missed, i apologize if so. overall, let us make sure all are safe and celebrate the outcome με strong coffee and future targets"},
    "Regex Words":        {"Text1": "today is our dragon boat festival in our chinese culture to celebrate it with all safe and great in our lives hope you too to enjoy it as my deepest wishes thank your message to show our words to the doctor as his next contract checking to all of us i got this message to see the approved message in fact i have received the message from the professor to show me this a couple of days ago i am very appreciated the full support of the professor for our springer proceedings publication", "Text2": "during our final discuss i told him about the new submission the one we were waiting since last autumn but the updates was confusing as it not included the full feedback from reviewer or maybe editor anyway i believe the team although bit delay and less communication at recent days they really tried best for paper and cooperation we should be grateful i mean all of us for the acceptance and efforts until the springer link came finally last week i think also kindly remind me please if the doctor still plan for the acknowledgments section edit he sending again because i didn t see that part final yet or maybe i missed i apologize if so overall let us make sure all are safe and celebrate the outcome strong coffee and future targets"},
    "Remove Stopwords":   {"Text1": "today dragon boat festival chinese culture celebrate safe great lives hope too enjoy deepest wishes thank your message show words doctor next contract checking us got message see approved message fact received message professor show couple days ago am appreciated full support professor springer proceedings publication", "Text2":"final discuss told him about new submission one were waiting autumn updates confusing included full feedback reviewer maybe editor anyway believe team although bit delay less communication recent days really tried best paper cooperation grateful mean us acceptance efforts springer link came finally week think also kindly remind please if doctor still plan acknowledgments section edit he sending again because i didn t see part final yet maybe missed apologize if overall let us make sure safe celebrate outcome strong coffee future targets"""},

}
# ─────────────────────────────────────────────────────────────────────────────
# 1) Φόρτωση πόρων & caching
# ─────────────────────────────────────────────────────────────────────────────
class EmbeddingManager:
    def __init__(self):
        print("Φόρτωση pre-trained embeddings…")
        self.w2v      = api.load("word2vec-google-news-300")
        self.glove    = api.load("glove-wiki-gigaword-50")
        self.fasttext = api.load("fasttext-wiki-news-subwords-300")

        print("Φόρτωση BERT…")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert      = BertModel.from_pretrained("bert-base-uncased", 
                                                   output_hidden_states=True)
        self.bert.eval()
        self._cache_bert = {}

    def get(self, word: str, kind: str) -> np.ndarray | None:
        if kind == "w2v":
            return self.w2v[word] if word in self.w2v else None
        if kind == "glove":
            return self.glove[word] if word in self.glove else None
        if kind == "fasttext":
            return self.fasttext[word] if word in self.fasttext else None
        if kind == "bert":
            if word in self._cache_bert:
                return self._cache_bert[word]
            # single-word batch
            inputs = self.tokenizer([word], return_tensors="pt", truncation=True)
            with torch.no_grad():
                hs = self.bert(**inputs).hidden_states[-1][0]  # (tokens, dim)
            emb = hs.mean(dim=0).cpu().numpy()
            self._cache_bert[word] = emb
            return emb
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 2) NLP Preprocessing pipelines
# ─────────────────────────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

def whitespace_split(text: str) -> list[str]:
    return text.lower().split()

def regex_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())

STOPWORDS = set(spacy.lang.en.stop_words.STOP_WORDS)
def remove_stopwords(text: str) -> list[str]:
    tokens = regex_tokenize(text)
    return [t for t in tokens if t not in STOPWORDS]

def lemmatize(text: str) -> list[str]:
    doc = nlp(text.lower())
    return [tok.lemma_ for tok in doc if tok.is_alpha]

def semantic_tree_vocab(text: str, depth: int = 2) -> list[str]:
    """Επέκταση λεξιλογίου με hypernyms από WordNet."""
    tokens = set(regex_tokenize(text))
    expanded = set(tokens)
    for t in tokens:
        for syn in wn.synsets(t):
            for path in syn.hypernym_paths():
                for h in path[:depth]:
                    expanded.update(h.lemma_names())
    return list(expanded)

PIPELINES = {
    "Whitespace Split": whitespace_split,
    "Regex Words":      regex_tokenize,
    "Remove Stopwords": remove_stopwords
    
}

CUSTOM_PIPELINES = {
    "Lemmatize": lemmatize,
    "WordNet+":  semantic_tree_vocab
}

# ─────────────────────────────────────────────────────────────────────────────
# 3) Μετρικές & Visualization
# ─────────────────────────────────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - cosine(a, b)

def text_mean_embedding(tokens: list[str],
                        emb_mgr: EmbeddingManager,
                        kind: str) -> np.ndarray | None:
    """
    Returns the mean (average) embedding vector for the list of tokens,
    or None if no token had an embedding.
    """
    embs = []
    for w in tokens:
        e = emb_mgr.get(w, kind)
        if e is not None:
            embs.append(e)
    if not embs:
        return None
    return np.mean(embs, axis=0)

def mean_similarity(orig_tokens: list[str],
                    recon_tokens: list[str],
                    emb_mgr: EmbeddingManager,
                    kind: str) -> float:
    """
    Compute cosine-similarity between the *mean* embeddings of
    orig_tokens vs recon_tokens.
    """
    vec_o = text_mean_embedding(orig_tokens, emb_mgr, kind)
    vec_r = text_mean_embedding(recon_tokens, emb_mgr, kind)
    if vec_o is None or vec_r is None:
        return float("nan")
    # cosine_sim = 1 - cosine_distance
    return 1.0 - cosine(vec_o, vec_r)


def compute_all_similarities(text_orig: str,
                             reconstructions: dict[str, dict[str,str]],
                             text_key: str,
                             emb_mgr: EmbeddingManager) -> pd.DataFrame:
    """
    Υπολογίζει μέσες cosine similarities για:
      1) κάθε pipeline του PIPELINES πάνω στα reconstructions[text_key]
      2) κάθε pipeline του CUSTOM_PIPELINES πάνω απευθείας στο text_orig

    Επιστρέφει DataFrame με στήλες: Pipeline, Embedding, CosineSim.
    """
    records = []
    # tokenize once the original
    orig_tokens = regex_tokenize(text_orig)

    # 1) Evaluate on the provided reconstructions
    for name, fn in PIPELINES.items():
        recon_text = reconstructions[name][text_key]
        recon_tokens = fn(recon_text)

        for emb in ["w2v", "glove", "fasttext", "bert"]:
            score = mean_similarity(orig_tokens, recon_tokens, emb_mgr, emb)
            records.append({
                "Pipeline":  name,
                "Embedding": emb,
                "CosineSim": score
            })

    # 2) Evaluate custom preprocessing directly on the original text
    for name, fn in CUSTOM_PIPELINES.items():
        recon_tokens = fn(text_orig)

        for emb in ["w2v", "glove", "fasttext", "bert"]:
            score = mean_similarity(orig_tokens, recon_tokens, emb_mgr, emb)
            records.append({
                "Pipeline":  name,
                "Embedding": emb,
                "CosineSim": score
            })

    return pd.DataFrame.from_records(records)



def visualize(df: pd.DataFrame, title:str):
    fig, ax = plt.subplots(figsize=(8,5))
    for emb in df["Embedding"].unique():
        subset = df[df["Embedding"]==emb]
        ax.plot(subset["Pipeline"], subset["CosineSim"], marker="o", label=emb)
    ax.set_title(title)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_ylim(0,1)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 4) Εκτέλεση για Text1 & Text2
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # instantiate once
    emb_mgr = EmbeddingManager()

    for label, orig in [("Text1", text1_orig), ("Text2", text2_orig)]:
        # pass emb_mgr as fourth argument
        df = compute_all_similarities(
            text_orig    = orig,
            reconstructions = reconstructions,
            text_key     = label,
            emb_mgr      = emb_mgr
        )

        print(f"\n=== Αποτελέσματα {label} ===")
        print(
            df
            .pivot(index="Pipeline", columns="Embedding", values="CosineSim")
            .round(4)
        )

        visualize(df, f"{label} Cosine-Sim Across Pipelines & Embeddings")


