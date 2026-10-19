# Data

Dataset: **Sentiment Labelled Sentences** (UCI Machine Learning Repository)
https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences

- Amazon, IMDB and Yelp subsets of sentences, labelled 0 (negative) or 1 (positive).
- Format: one `sentence<TAB>label` per line, no header (e.g. `amazon_cells_labelled.txt`).
- Citation: Kotzias et al., "From Group to Individual Labels using Deep Features", KDD 2015.
- Check the licence on the dataset page before redistributing. Raw data is not committed here.

`python -m src.train` downloads the Amazon subset into `data/raw/` automatically. If you are offline,
download the zip yourself and place `amazon_cells_labelled.txt` in `data/raw/`.
