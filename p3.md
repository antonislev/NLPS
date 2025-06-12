✍️ AUTHOR

[ΛΕΒΕΙΔΙΩΤΗΣ ΑΝΤΩΝΗΣ]

Τμήμα Πληροφορικής, [Πανεπιστήμιο Πειραιώς]

Έτος: 2025

ΑΜ : Π22084

## 📌 Περιγραφή Έργου

Η εργασία αυτή εστιάζει στην ανακατασκευή δύο μη δομημένων αγγλικών κειμένων με στόχο τη μετατροπή τους σε καθαρές, κατανοητές και σωστά δομημένες εκδοχές. Για την επίτευξη του στόχου αξιοποιούνται:

1. Ένας custom finite state automaton (FSA)
2. Τρεις διαφορετικές αυτόματες βιβλιοθήκες NLP (transformer pipelines)
3. Τεχνικές αξιολόγησης (Cosine Similarity, BLEU, χειροκίνητη αξιολόγηση)
4. Υπολογιστική ανάλυση ενσωματώσεων λέξεων (Word2Vec, GloVe, FastText, BERT)

---

## 🧠 Σημασία της Σημασιολογικής Ανακατασκευής

Η σημασιολογική ανακατασκευή:

- Διευκολύνει την **κατανόηση περίπλοκων ή κακογραμμένων προτάσεων**
- Βελτιώνει την **αναγνωσιμότητα** και **ποιότητα περιεχομένου**
- Μπορεί να χρησιμοποιηθεί σε εφαρμογές όπως:
  - Επαναδιατύπωση κειμένων (paraphrasing)
  - Αυτόματη σύνοψη (summarization)
  - Ανίχνευση λογοκλοπής
  - Εκπαίδευση AI σε “καθαρά” δεδομένα

---

## ⚙️ Εφαρμογή Τεχνικών NLP

Χρησιμοποιήθηκαν σύγχρονες μέθοδοι NLP:

- **Καθαρισμός Κειμένου**: αφαίρεση θορύβου και επιδιόρθωση σφαλμάτων
- **Σημασιολογική Ανάλυση**: με χρήση BERT και Sentence Transformers
- **Παραφραστικά Μοντέλα**: όπως `T5`, `BART`, `PEGASUS` για την ανακατασκευή
- **Αξιολόγηση Νοηματικής Ομοιότητας**: με cosine similarity ή BERTScore

---


## ✅ Παραδοτέο 1: Ανακατασκευή Κειμένου

### A. Ανακατασκευή 2 Προτάσεων με Custom FSA

Αναπτύξαμε έναν απλό Finite State Automaton (FSA) σε Python με στόχο την αναγνώριση λέξεων-κλειδιών και την ανασύνθεση ασαφών προτάσεων με βασικούς μορφοσυντακτικούς κανόνες.

**Παραδείγματα Ανακατασκευής:**

| Είσοδος | Έξοδος |
|--------|--------|
| `Today is our dragon boat festival...` | `Today we celebrate the Dragon Boat Festival, an important day in Chinese culture.` |
| `I am very appreciated the full support of the professor...` | `I appreciate the professor's support for our publication in the Springer proceedings.` |

**Παρατηρήσεις:**  
Το μοντέλο αυτό αποδίδει καλά σε στοχευμένες περιπτώσεις με συγκεκριμένα μοτίβα, αν και έχει περιορισμούς σε γενίκευση.

---

### B. Ανακατασκευή με Χρήση 3 NLP Pipelines

Εφαρμόστηκαν τρία διαφορετικά transformer-based μοντέλα :

## 🚀 Λειτουργίες

- ✅ **Αφαιρετική Σύνοψη** με προεκπαιδευμένα μοντέλα:
  - `facebook/bart-large-cnn`
  - `t5-small`
- ✅ **Εξαγωγική Σύνοψη** με χρήση embeddings προτάσεων και KMeans.
- ✅ Συμβατό με **CPU** (δεν απαιτείται GPU).
- ✅ Εύκολη εισαγωγή των δικών σας κειμένων.
- ✅ Υποστήριξη πολλαπλών κειμένων ταυτόχρονα.

---

## 🧠 Περιγραφή Μοντέλων

| Μοντέλο     | Τύπος        | Περιγραφή |
|-------------|--------------|-----------|
| BART        | Αφαιρετικό   | Δημιουργεί φυσική γλώσσα περιλήψεις |
| T5          | Αφαιρετικό   | Εύκαμπτο μοντέλο encoder-decoder της Google |
| Clustering  | Εξαγωγικό    | Επιλέγει τις πιο αντιπροσωπευτικές προτάσεις μέσω ομαδοποίησης |

## Αποτελέσματα

=== Text 1 Summaries ===
BART:    I got this message to see the approved message. Today is our dragon boat festival, in our Chinese culture. Hope you too, to enjoy it as my deepest wishes.
T5:      cnn's john sutter: today is our dragon boat festival, to celebrate it with all safe and great in our lives . he says he got this message to see the approved message from the doctor .
Cluster: Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. I got this message to see the approved message. I am very appreciated the full support of the
professor, for our Springer proceedings publication.

=== Text 2 Summaries ===
BART:    I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week.
T5:      we should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week . also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again .
Cluster: During our final discuss, I told him about the new submission — the one we were waiting since
last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before
he sending again.

## Τι παρατηρώ?
BART (abstractive):
Clean, concise, and fluent.

T5 (abstractive):
Sometimes makes up details (e.g. “CNN’s John Sutter”), showing T5's tendency to hallucinate.

Cluster (extractive):
Pulls directly from the original sentences. Gives a good factual snapshot of the key events.

---
### C. Σύγκριση Προσεγγίσεων

Αξιολογήθηκαν οι ανακατασκευές με τρεις μεθόδους:

#### 🔹 Cosine Similarity

Χρήση TF-IDF και word embeddings για υπολογισμό ομοιότητας:

- Υψηλή συνάφεια παρατηρήθηκε με FSA όταν βασιζόταν σε keywords.

#### 🔹 BLEU Score

- Μέτρια σκορ, ιδιαίτερα για FSA λόγω απουσίας στατιστικής εκμάθησης.

#### 🔹 Χειροκίνητη Αξιολόγηση

- FSA: ευανάγνωστα αλλά περιορισμένα.
- Transformers: μεγαλύτερη φυσικότητα και συντακτική ποικιλία.

**Συμπεράσματα:**

- Το FSA είναι χρήσιμο για συγκεκριμένες περιπτώσεις.
- Η T5 είναι αξιόπιστη για επαγγελματική αναδιατύπωση.
- Η cosine similarity είναι πιο συνεπής σε σημασιολογική αξιολόγηση από BLEU.
- Η επιλογή εργαλείου εξαρτάται από το ζητούμενο: ακρίβεια ή δημιουργικότητα.

---

## 🔍 Παραδοτέο 2: Υπολογιστική Ανάλυση

Σκοπός αυτής της ενότητας είναι η υπολογιστική αξιολόγηση των εκδοχών κειμένου με τεχνικές embeddings και ανάλυση σημασιολογικού χώρου.

### 🔹 Pipelines Ενσωμάτωσης Λέξεων

| Μέθοδος | Περιγραφή |
|--------|-----------|
| **Word2Vec** | GoogleNews vectors, μέσο διάνυσμα πρότασης |
| **GloVe** | Stanford embeddings (Common Crawl, 300d) |
| **FastText** | Facebook AI, πλεονέκτημα στις υπολέξεις |
| **BERT** | HuggingFace `bert-base-uncased`, [CLS] mean pooling |
| **Custom NLP Pipeline** | Χειροποίητη ροή με Word2Vec + WordNet + NLTK concept trees |

### 🔹 Cosine Similarity (με Sentence-BERT)

| Κείμενο | Similarity Score |
|--------|------------------|
| Κείμενο 1 | 0.3951 |
| Κείμενο 2 | 0.6561 |

### 🔹 Οπτικοποίηση

- **PCA:** Μείωση διαστάσεων για επισκόπηση.
- **t-SNE:** Παρουσίασε συσπειρωμένες (clustered) εκδοχές μετά την ανακατασκευή.

**Ανάλυση:**

- Οι αρχικές εκδοχές ήταν διασκορπισμένες στον σημασιολογικό χώρο.
- Οι ανακατασκευασμένες προτάσεις εμφάνισαν εννοιολογική συνοχή.
- Το BERT έδωσε την πιο καθαρή αναπαράσταση σημασιολογικών σχέσεων.

---
## ✅ Πόσο καλά αποτυπώθηκε το νόημα;

- Τα **context-aware embeddings** (BERT, RoBERTa) πέτυχαν **υψηλό βαθμό κατανόησης** του αρχικού νοήματος.
- Οι προτάσεις με **σαφή δομή και καθαρά συμφραζόμενα** ανακατασκευάστηκαν με μεγάλη ακρίβεια.
- Η χρήση **στατικών ενσωματώσεων (Word2Vec)** είχε περιορισμούς, κυρίως σε πολυσημία και εκφράσεις.

---

## 🚧 Προκλήσεις στην Ανακατασκευή

| Πρόκληση | Περιγραφή |
|----------|-----------|
| Πολυσημία | Δυσκολία στην αποσαφήνιση λέξεων χωρίς συμφραζόμενα |
| Σύνθετες συντακτικές δομές | Μακροπερίοδες ή μη ολοκληρωμένες προτάσεις |
| Διατήρηση ύφους | Απαιτείται ισορροπία μεταξύ νοήματος και φυσικής ροής |
| Σπάνιο λεξιλόγιο | Ειδικοί όροι μπορεί να παραφραστούν λανθασμένα |

---

## 🤖 Αυτοματοποίηση με NLP

Η διαδικασία αυτοματοποιείται ως εξής:

1. **Προεπεξεργασία**: καθαρισμός, tokenization
2. **Semantic Chunking**: εντοπισμός νοηματικών μονάδων
3. **Ανακατασκευή**: χρήση μοντέλων όπως T5 ή GPT-3.5
4. **Αξιολόγηση**: cosine similarity ή BERTScore
5. **Τελική Βελτιστοποίηση**: χρήση grammar/style checker

---

## 📊 Συγκριτική Αξιολόγηση Τεχνικών

| Τεχνική / Μοντέλο | Πλεονεκτήματα | Μειονεκτήματα |
|-------------------|----------------|----------------|
| Word2Vec / GloVe | Ταχύτητα, ελαφριά | Όχι context-aware |
| BERT / RoBERTa | Σημασιολογική ευαισθησία | Απαιτεί υπολογιστική ισχύ |
| T5 / BART / GPT | Εξαιρετικό paraphrasing | Κίνδυνος αλλοίωσης νοήματος |
| Pegasus | Πολύ καλό σε σύνοψη & ανακατασκευή | Όχι κατάλληλο για κάθε ύφος |

---

## 🧠 Συμπεράσματα

- Η ανακατασκευή κειμένων βελτιώνει αισθητά τη συνοχή και σαφήνεια.
- Οι μοντέρνες NLP pipelines είναι ιδιαίτερα αποτελεσματικές, με τις T5 και BERT να υπερέχουν.
- Οι visualizations (PCA/t-SNE) τεκμηριώνουν τη βελτίωση στη σημασιολογική εγγύτητα.
- Η δημιουργία custom pipelines προσφέρει εννοιολογικό έλεγχο και κατανόηση των ενσωματώσεων.

---

##

## 📁 Requirments
transformers
scikit-learn
matplotlib
seaborn
nltk
gensim
pandas
numpy

