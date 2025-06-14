**✍️AUTHOR**
**ΛΕΒΕΙΔΙΩΤΗΣ ΑΝΤΩΝΗΣ** (ΑΜ: Π22084)
Τμήμα Πληροφορικής, Πανεπιστήμιο Πειραιώς
Έτος: 2025

---

## 📌 Περίληψη

Η εργασία αυτή παρουσιάζει ένα ολοκληρωμένο πλαίσιο **σημασιολογικής ανακατασκευής κειμένου**. Συνδυάζονται rule‑based αυτοματοποιημένοι κανόνες (DFA), παραδοσιακά NLP pipelines (spaCy, NLTK, Gensim) και σύγχρονες ενσωματώσεις λέξεων (Word2Vec, GloVe, FastText, BERT). Πραγματοποιείται σύγκριση ως προς λεξιλογική πιστότητα (TTR, Jaccard), σημασιολογική συνάφεια (cosine similarity) και οπτικοποίηση μετατοπίσεων μέσω PCA/t‑SNE. Τα αποτελέσματα επιβεβαιώνουν τη διατήρηση νοήματος και αναδεικνύουν trade‑offs μεταξύ καθαρότητας και γενίκευσης.

---

## 1. Εισαγωγή

Η **σημασιολογική ανακατασκευή κειμένου** στοχεύει στην παραγωγή καθαρών, συνεκτικών και ακριβών εκδόσεων πρωτογενών κειμένων, διατηρώντας παράλληλα το αρχικό νόημα. Μέσω τεχνικών NLP αφαιρούμε θόρυβο, γραμματικές ανωμαλίες και πλεονασμούς, ενώ με embeddings αποτυπώνουμε σημασιολογικές σχέσεις.

**Συμβολή εργασίας**:

1. Ορισμός και υλοποίηση **DFA** για δύο παράδειγμα προτάσεις (DFA‑based reconstruction).
2. Σύγκριση **τριών Python pipelines** (spaCy, NLTK, Gensim) για ολόκληρα κείμενα.
3. Εφαρμογή **Word2Vec, GloVe, FastText, BERT embeddings** και custom flows (Whitespace, Regex, Stopwords) με υπολογισμό cosine similarity.
4. Οπτικοποίηση σημασιολογικών μετατοπίσεων μέσω **PCA** και **t‑SNE**.

---

## 2. Μεθοδολογία

### 2.1. Ανακατασκευή δύο προτάσεων (DFA)

Χρησιμοποιήθηκε ένας **Deterministic Finite Automaton** για τις προτάσεις:

* **A**: "The quick brown fox"
* **B**: "Jump over lazy dog"

**Ορισμός DFA**:

* Καταστάσεις: `q0` (start), `q1`–`q4` (A), `q5`–`q8` (B).
* Αλφάβητο: οι λέξεις κάθε πρότασης.
* Accept states: `q4` (τέλος A), `q8` (τέλος B).

```python
transitions = {
    ("q0", "The"): "q1",
    ("q1", "quick"): "q2",
    ("q2", "brown"): "q3",
    ("q3", "fox"): "q4",
    ("q0", "Jump"): "q5",
    ("q5", "over"): "q6",
    ("q6", "lazy"): "q7",
    ("q7", "dog"): "q8"
}
```

Μέσω **BFS** εξάγονται οι μοναδικές διαδρομές προς `q4` και `q8`, ανακατασκευάζοντας ακριβώς τις δύο προτάσεις.

### 2.2. Τρία Python pipelines για ολόκληρα κείμενα

Εφαρμόστηκαν στα δύο πρωτογενή κείμενα (Text1, Text2) οι εξής ροές:

1. **spaCy Lemmatization**: tokenization → αφαίρεση stopwords → λεμματοποίηση.
2. **NLTK RegexpTokenizer**: regex-based tokenization → stopwords → WordNet lemmatizer.
3. **Gensim simple\_preprocess**: deaccented tokenization → Gensim stopword list.

Κατόπιν επανασυντέθηκαν τα tokens σε συνεχή κείμενα για περαιτέρω ανάλυση.

### 2.3. Metrics σύγκρισης

Για κάθε pipeline και κείμενο υπολογίστηκαν:

* **Total tokens** & **Unique tokens**
* **Type–Token Ratio (TTR)** = Unique / Total
* **Jaccard Similarity** vs. αρχικό λεξιλόγιο

Τα metrics ομαδοποιήθηκαν σε DataFrame και παρουσιάστηκαν με γραμμικά διαγράμματα.

---

## 3. Ενσωματώσεις Λέξεων & Semantic Analysis

### 3.1. Προεκπαιδευμένα models

* **Word2Vec** (Google News, 300d)
* **GloVe** (Wiki‑Gigaword, 50d)
* **FastText** (Wiki‑Subwords, 300d)
* **BERT** (bert-base-uncased, mean-pooled)

### 3.2. Υπολογισμός cosine similarity

Για κάθε κείμενο (πριν/μετά) υπολογίστηκε το μέσο embedding vector και η **cosine similarity**:

```python
vec_orig = mean_embedding(orig_tokens, model)
vec_recon = mean_embedding(recon_tokens, model)
score = 1 - cosine(vec_orig, vec_recon)
```

### 3.3. Custom NLP flows

Επιπλέον examiner pipelines:

* Whitespace Split
* Regex Tokenize
* Remove Stopwords
* Lemmatize (spaCy)
* WordNet + Hypernyms

### 3.4. Οπτικοποίηση (PCA & t‑SNE)

Αρχικά εφαρμόστηκε **PCA→2D**, στη συνέχεια **t‑SNE→2D** (perplexity=30, init='random').

---

## 4. Πειράματα & Αποτελέσματα

### 4.1. Ανακατασκευές

**Παράδειγμα A**:

* Πριν: "Today is our dragon boat festival"
* DFA:  "Today is our dragon boat festival"

**Text1 (spaCy)**:

* Πριν: Today is our dragon boat festival, in our Chinese culture, to celebrate...
* Μετά: today dragon boat festival chinese culture celebrate safe great lives hope

### 4.2. Metrics

| Pipeline | Text  | Total | Unique | TTR   | Jaccard |
| -------- | ----- | ----- | ------ | ----- | ------- |
| spaCy    | Text1 | 37    | 33     | 0.892 | 0.524   |
| NLTK     | Text2 | 59    | 57     | 0.966 | 0.525   |
| Gensim   | Text1 | 37    | 33     | 0.892 | 0.524   |

### 4.3. Cosine Similarities

| Model    | Text1 vs Recon | Text2 vs Recon |
| -------- | -------------- | -------------- |
| Word2Vec | 0.985          | 0.978          |
| GloVe    | 0.982          | 0.975          |
| FastText | 0.987          | 0.980          |
| BERT     | 0.991          | 0.989          |

---

## 5. Συζήτηση

Οι υψηλές similarity scores (>0.97) επιβεβαιώνουν τη διατήρηση του νοήματος ακόμη και μετά από έντονο preprocessing.
Οι pipelines με stopword removal και hypernyms παρουσίασαν μικρότερες τιμές (\~0.90–0.92) εξαιτίας λεξιλογικών απωλειών.
Η οπτικοποίηση PCA/t‑SNE δείχνει σταθερές τοπικές δομές: οι λέξεις διατηρούν τις ομάδες τους (π.χ. ζώα, χρώματα).

---

## 6. Συμπεράσματα & Μελλοντικές Εργασίες

* Δεν υπάρχει μία ιδανική pipeline· η επιλογή εξαρτάται από τον στόχο: **πιστότητα** vs. **γενίκευση**.
* Συνδυαστικά rule‑based + transformer pipelines μπορούν να ενισχύσουν contextual paraphrasing.
* Μελλοντικά: fine‑tuned seq2seq models (BART, T5) σε παράλληλα corpus για end‑to‑end παραφράσεις.

---

## 📁 Requirements

```text
numpy>=1.18.5
scipy>=1.7.0
gensim>=4.3.3
scikit-learn>=1.2
matplotlib>=3.5
torch>=1.13
transformers>=4.30
nltk
seaborn
pandas
```
