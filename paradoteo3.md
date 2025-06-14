✍️ **Συγγραφέας**
**ΛΕΒΕΙΔΙΩΤΗΣ ΑΝΤΩΝΗΣ** (ΑΜ: Π22084)
Τμήμα Πληροφορικής, Πανεπιστήμιο Πειραιώς
Έτος: 2025

---

## 📌 Περίληψη

Η εργασία αυτή παρέχει μια αναλυτική προσέγγιση στη σημασιολογική ανακατασκευή κειμένων, συνδυάζοντας rule-based στρατηγικές και σύγχρονες τεχνικές NLP, όπως pipelines spaCy, NLTK, Gensim και embeddings Word2Vec, GloVe, FastText, BERT. Πραγματοποιείται διεξοδική σύγκριση με βάση λεξιλογικά metrics (TTR, Jaccard), σημασιολογική συνάφεια (cosine similarity) και οπτικοποιήσεις PCA/t-SNE. Τα αποτελέσματα αναδεικνύουν trade-offs μεταξύ λεξιλογικής πιστότητας και σημασιολογικής γενίκευσης.

---

## 1. Εισαγωγή

Η **σημασιολογική ανακατασκευή κειμένου** είναι κρίσιμη στην αποσαφήνιση, τον καθαρισμό και τη βελτίωση της ποιότητας των κειμένων, διατηρώντας παράλληλα τη βασική σημασιολογική τους υπόσταση. Με την αξιοποίηση τεχνικών NLP, όπως λεμματοποίηση και embeddings, επιτυγχάνεται η αφαίρεση θορύβου, γραμματικών ασαφειών και πλεονασμών, καθιστώντας τα κείμενα πιο κατανοητά και κατάλληλα για περαιτέρω αναλύσεις.

---

## 2. Μεθοδολογία

### 2.1. Στρατηγικές Ανακατασκευής

#### A. Γραμματική/Αξιώματα (DFA)

Εφαρμόστηκε ένας **Deterministic Finite Automaton (DFA)** για ακριβή ανακατασκευή δύο προτάσεων:

* Πρόταση Α: "The quick brown fox"
* Πρόταση Β: "Jump over lazy dog"

Οι καταστάσεις του DFA διαμορφώθηκαν έτσι ώστε να είναι μονοσήμαντες, προσφέροντας μια σαφή και επαναλήψιμη διαδικασία ανακατασκευής.

#### B. Γλωσσικοί Κανόνες (Pipelines)

Χρησιμοποιήθηκαν τρία διαφορετικά pipelines:

* spaCy (λεμματοποίηση, stopwords)
* NLTK (RegexpTokenizer, WordNet Lemmatizer)
* Gensim (simple\_preprocess)

Τα pipelines στόχευσαν σε πλήρη κείμενα, εφαρμόζοντας διαφορετικές στρατηγικές tokenization και preprocessing για αναλυτική σύγκριση.

#### C. Σημασιολογική Ανακατασκευή (Embeddings)

Αξιοποιήθηκαν embeddings από Word2Vec, GloVe, FastText και BERT για την αποτύπωση σημασιολογικών σχέσεων μεταξύ των λέξεων και τη διατήρηση του νοήματος μετά από preprocessing.

### 2.2. Υπολογιστικές Τεχνικές

Για τη σημασιολογική ανάλυση εφαρμόστηκαν:

* Υπολογισμός Cosine similarity μεταξύ των μέσων διανυσμάτων embeddings των πρωτογενών και των ανακατασκευασμένων κειμένων.
* Οπτικοποίηση των αποτελεσμάτων με PCA και t-SNE, για να φανεί η σημασιολογική σταθερότητα.

---

## 3. Πειράματα & Αποτελέσματα

### Παραδείγματα Ανακατασκευής

**Παράδειγμα κειμένου A (πριν/μετά):**

* Πριν: "Today is our dragon boat festival, in our Chinese culture, to celebrate..."
* Μετά (spaCy): "today dragon boat festival chinese culture celebrate safe great lives hope"

### Metrics και Ανάλυση (Παραδοτέο 2)

| Pipeline | Text  | Total Tokens | Unique Tokens | TTR   | Jaccard Similarity |
| -------- | ----- | ------------ | ------------- | ----- | ------------------ |
| spaCy    | Text1 | 37           | 33            | 0.892 | 0.524              |
| NLTK     | Text2 | 59           | 57            | 0.966 | 0.525              |
| Gensim   | Text1 | 37           | 33            | 0.892 | 0.524              |

| Model    | Cosine Similarity (Text1) | Cosine Similarity (Text2) |
| -------- | ------------------------- | ------------------------- |
| Word2Vec | 0.985                     | 0.978                     |
| GloVe    | 0.982                     | 0.975                     |
| FastText | 0.987                     | 0.980                     |
| BERT     | 0.991                     | 0.989                     |

---

## 4. Συζήτηση

Τα embeddings απέδωσαν υψηλή διατήρηση νοήματος (scores > 0.97), γεγονός που επιβεβαιώνει την αποτελεσματικότητά τους. Ωστόσο, η μεγαλύτερη πρόκληση ήταν η ισορροπία μεταξύ αφαίρεσης λέξεων (stopwords, hypernyms) και διατήρησης νοήματος.

Η αυτοματοποίηση μπορεί να ενισχυθεί σημαντικά μέσω transformer-based μοντέλων, καθώς παρέχουν δυναμική σημασιολογική κατανόηση που ξεπερνά τις παραδοσιακές μεθόδους.

Υπήρξαν αξιοσημείωτες διαφορές ποιότητας μεταξύ των pipelines. Για παράδειγμα, η spaCy προσέφερε υψηλή λεξιλογική καθαρότητα, ενώ η NLTK παρουσίασε μεγαλύτερη λεξιλογική ποικιλία αλλά μικρότερη πιστότητα σημασίας.

---

## 5. Συμπέρασμα

Η εργασία ανέδειξε πως η επιλογή τεχνικών ανακατασκευής είναι συνάρτηση του επιθυμητού αποτελέσματος (πιστότητα έναντι γενίκευσης). Η ενσωμάτωση transformer-based μοντέλων φαίνεται ιδιαίτερα υποσχόμενη για πλήρη αυτοματοποίηση της σημασιολογικής ανακατασκευής. Μελλοντική κατεύθυνση αποτελεί η αξιοποίηση seq2seq μοντέλων όπως τα BART και T5 σε ειδικευμένα corpus για την περαιτέρω ενίσχυση της αποτελεσματικότητας και ακρίβειας της ανακατασκευής.


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
