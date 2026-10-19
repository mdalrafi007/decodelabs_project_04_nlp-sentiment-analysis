# Models

`python -m src.train` writes `sentiment_model.joblib` and `model_metadata.json` here.

- The `.joblib` file is a pickled scikit-learn pipeline (preprocessing + TF-IDF + classifier).
- Pickles can break across scikit-learn/NLTK versions and can execute code when loaded. Only load files you created.
- Load it from the project root (`python -m src.predict`) so the `src` package is importable.
