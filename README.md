NLP Project: Semantic Text Reconstruction

Λεβειδιώτης Αντώνης (ΑΜ: Π22084)

Τμήμα Πληροφορικής, Πανεπιστήμιο Πειραιώς

Περίληψη

Σε αυτήν την εργασία αναπτύσσω και συγκρίνω διάφορες μεθόδους σημασιολογικής ανακατασκευής κειμένου. Οι κύριες συνιστώσες περιλαμβάνουν:

DFA-based Reconstruction: καθορισμός deterministic finite automaton για δύο προτάσεις-παράδειγμα.

NLP Pipelines: εφαρμόζω spaCy, NLTK και Gensim για επεξεργασία ολόκληρων κειμένων.

Embeddings Analysis: φόρτωση Word2Vec, GloVe, FastText, BERT και σύγκριση των embeddings πριν και μετά την ανακατασκευή.

Visualization: οπτικοποίηση με PCA & t-SNE για ανάδειξη σημασιολογικών μετατοπίσεων.

Περιεχόμενα

dfa_reconstruct.py: υλοποίηση DFA για δύο προτάσεις.

pipeline_reconstruct.py: τρία Python pipelines (spaCy, NLTK, Gensim).

embeddings_analysis.py: φόρτωση embeddings, υπολογισμός cosine similarity.

vis.py: οπτικοποίηση PCA & t-SNE των embeddings.

requirements.txt: απαραίτητες βιβλιοθήκες.

Οδηγίες Εκτέλεσης

Δημιουργήστε virtual environment:

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate.ps1 # Windows PowerShell

Εγκαταστήστε τις απαιτήσεις:

pip install -r requirements.txt

Τρέξτε τα scripts:

python dfa_reconstruct.py

python pipeline_reconstruct.py

python embeddings_analysis.py

python vis.py

Στόχοι

Αξιολόγηση πώς διαφορετικές τεχνικές NLP επιδρούν στο νόημα.

Ποσοτικές μετρήσεις μέσω cosine similarity, TTR, Jaccard.

Ποιοτικές παρατηρήσεις μέσω οπτικοποιήσεων PCA/t-SNE.

Επικοινωνία

Email: your.email@example.com

22 Ιουνίου 2025

