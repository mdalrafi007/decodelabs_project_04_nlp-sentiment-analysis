# NLP & Sentiment Analysis (DecodeLabs Data Science Project 4)

Classify product reviews as **Positive** or **Negative** using an NLTK preprocessing pipeline,
TF-IDF features, and two classifiers (Multinomial Naive Bayes and Linear SVM) compared fairly.

> **Status:** the code is complete, but this README contains no metrics on purpose. Run the
> project, then use the numbers in `reports/results/RESULTS.md` (generated from your actual run).

## DecodeLabs requirements covered
| Requirement | Where |
|---|---|
| Tokenisation, stop-word removal, lemmatisation (NLTK) | `src/preprocessing.py` |
| TF-IDF vectors | `src/model.py` (`TfidfVectorizer`) |
| Naive Bayes or SVM classifier (both are trained) | `src/model.py`, `src/train.py` |
| Predict a new review as Positive/Negative | `src/predict.py` |

## Dataset
UCI *Sentiment Labelled Sentences*, Amazon subset (`amazon_cells_labelled.txt`): sentences labelled
0 = Negative, 1 = Positive. Source: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
(Kotzias et al., KDD 2015). See `data/README.md`. The file is small, so test-set scores vary noticeably
with the split; treat them as estimates, not guarantees.

## Setup on Android with Termux
1. Install Termux (from F-Droid or GitHub releases) and run `termux-setup-storage`.
2. Copy the project zip from Downloads and unzip:
   ```bash
   cp ~/storage/downloads/nlp-sentiment-analysis.zip ~
   cd ~ && pkg install -y unzip && unzip nlp-sentiment-analysis.zip && cd nlp-sentiment-analysis
   ```
3. Install everything (needs internet; the scikit-learn step can take a while if it must compile):
   ```bash
   bash setup_termux.sh
   ```
4. Train, evaluate, and save the model (downloads the dataset automatically):
   ```bash
   python -m src.train
   ```
5. Predict:
   ```bash
   python -m src.predict "The battery died after two days, not happy"
   python -m src.predict --demo     # illustrative examples
   python -m src.predict            # interactive prompt
   ```
6. Run the tests:
   ```bash
   python -m pytest -q
   ```
Always run commands from the project root (`nlp-sentiment-analysis/`).

**Other systems:** `python -m venv .venv`, activate it, `pip install -r requirements.txt`, then the same commands.
Python 3.10+ recommended. No GPU or API key needed.

**Offline dataset:** download the zip from the UCI page, extract `amazon_cells_labelled.txt` into
`data/raw/`, then run `python -m src.train --no-download`.

## What the pipeline does
1. **Normalise:** lower-case, remove HTML tags and URLs, expand contractions (`didn't` -> `did not`).
   Numbers are kept; punctuation is dropped at tokenisation (the vectorizer ignores it anyway).
2. **Tokenise** with NLTK `word_tokenize`, then **POS-tag** each token.
3. **Stop words:** NLTK English list **minus** negations (`no, nor, not, never, ...`) and `very, too, but`.
   Removing "not" would turn "not good" into "good", which flips the sentiment.
4. **Lemmatise** with NLTK `WordNetLemmatizer`, guided by the POS tag so verbs and adjectives reduce correctly.
5. **TF-IDF:** each review becomes a sparse vector. Term frequency (how often a word appears in the review)
   is multiplied by inverse document frequency (how rare it is across reviews), so common filler gets low
   weight and distinctive words get high weight. Settings: unigrams+bigrams (captures "not good"),
   `max_features=10000`, `max_df=0.95`, `sublinear_tf=True`, `min_df` tuned (1 or 2). Output stays sparse.
6. **Classifiers:** Multinomial Naive Bayes (fast probabilistic baseline that suits word-weight features)
   and Linear SVM (finds a separating hyperplane in the high-dimensional TF-IDF space).

## Evaluation design (leakage-safe)
- Stratified 80/20 split, seed 42. The test set is untouched until the final evaluation.
- Preprocessing + TF-IDF + classifier live in one scikit-learn `Pipeline`, so TF-IDF is fitted only on
  training text, and inside each cross-validation fold separately.
- Small grid search with 5-fold CV on the **training split only**; the final model is the one with the
  best cross-validated F1. The test set is never used to choose.
- Reported on the test set: accuracy, precision, recall, F1, ROC-AUC (positive class = Positive), confusion
  matrices, classification reports. LinearSVC uses decision scores for ROC-AUC, not fake probabilities.
- Outputs: `reports/figures/` (class balance, review lengths, confusion matrices, model comparison),
  `reports/results/` (`RESULTS.md`, `metrics.json`, `errors.csv`, `top_features.csv`).

## Repository structure
```
src/            data_loader, preprocessing, model, eda, evaluate, train, predict
tests/          preprocessing and pipeline tests
notebooks/      sentiment_analysis.ipynb (walkthrough; see note below)
reports/        generated figures and results
models/         saved pipeline (generated, git-ignored)
setup_termux.sh Termux installer
```

## Prediction notes
`predict_review` rejects empty/non-string input and text with no usable words. Naive Bayes reports
`P(Positive)` (may be poorly calibrated); the SVM reports a decision score (> 0 = Positive) that is
**not** a probability. Load only `.joblib` files you created: pickles can execute code and may break
across library versions. Versions used are recorded in `models/model_metadata.json`.

## Notebook
`notebooks/sentiment_analysis.ipynb` reuses the `src` modules. It is easiest in Google Colab or Kaggle
(upload the project folder, `%cd` into it). It has **not been executed**; run it top to bottom and write
your own observations in the markdown cells after each output.

## Known limitations
Small dataset; short single-sentence reviews; sarcasm, mixed sentiment and spelling errors are hard for
bag-of-words models; features reflect associations in this data, not universal word meanings; results are
for English product reviews only.

## Troubleshooting
- `NLTK resources are missing`: run `python -m nltk.downloader punkt_tab punkt stopwords wordnet averaged_perceptron_tagger_eng averaged_perceptron_tagger`.
- `pip install scikit-learn` fails in Termux: retry after `pkg install -y clang make pkg-config`, or run the same
  project in Google Colab.
- `ModuleNotFoundError: src`: you are not in the project root.

## Future improvements
Larger datasets (`--subset all` adds IMDB and Yelp), ComplementNB for imbalanced data, probability
calibration for the SVM, error-driven feature work, a small Streamlit demo.

## Acknowledgements
DecodeLabs Industrial Training Kit (Project 4) and the UCI Machine Learning Repository dataset.
