import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import gensim.downloader as api
import torch
from transformers import BertTokenizer, BertModel

# ----------------------------
# 1) Φόρτωση των embeddings
# ----------------------------
w2v = api.load("word2vec-google-news-300")
glove = api.load("glove-wiki-gigaword-50")
fasttext = api.load("fasttext-wiki-news-subwords-300")

tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
model_bert = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
model_bert.eval()

# ----------------------------
# 2) Ορισμός κειμένων & reconstructions
# ----------------------------
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

reconstructions = {
    "Whitespace Split": {
        "Text1": text1_orig.lower().split(),
        "Text2": text2_orig.lower().split()
    },
    "Regex Words": {
        "Text1": re.findall(r"[a-zA-Z]+", text1_orig.lower()),
        "Text2": re.findall(r"[a-zA-Z]+", text2_orig.lower())
    }
}

# ----------------------------
# 3) Βοηθητικές συναρτήσεις
# ----------------------------
def tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())

def get_embedding(word, kind):
    if kind == "w2v":
        return w2v[word] if word in w2v else None
    if kind == "glove":
        return glove[word] if word in glove else None
    if kind == "fasttext":
        return fasttext[word] if word in fasttext else None
    if kind == "bert":
        inputs = tokenizer_bert(word, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model_bert(**inputs)
        vec = outputs.hidden_states[-1][0].mean(dim=0).cpu().numpy()
        return vec
    return None

# ----------------------------
# 4) Οπτικοποίηση PCA & t-SNE
# ----------------------------
def visualize(kind, pipeline, tokens):
    # συλλογή embeddings
    embs = [get_embedding(w, kind) for w in tokens]
    embs = [e for e in embs if e is not None]
    data = np.vstack(embs)

    # PCA
    coords_pca = PCA(n_components=2).fit_transform(data)
    plt.figure()
    plt.scatter(coords_pca[:,0], coords_pca[:,1], marker='o')
    plt.title(f"{pipeline} — PCA ({kind})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

    # t-SNE
    coords_tsne = TSNE(n_components=2, random_state=0, init="random", learning_rate="auto").fit_transform(data)
    plt.figure()
    plt.scatter(coords_tsne[:,0], coords_tsne[:,1], marker='o')
    plt.title(f"{pipeline} — t-SNE ({kind})")
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.tight_layout()
    plt.show()

# ----------------------------
# 5) Εκτέλεση για κάθε συνδυασμό
# ----------------------------
embedding_types = ["w2v", "glove", "fasttext", "bert"]
for kind in embedding_types:
    for pipeline_name, texts in reconstructions.items():
        for label, tokens in texts.items():
            viz_title = f"{label}"
            visualize(kind, viz_title, tokens)


