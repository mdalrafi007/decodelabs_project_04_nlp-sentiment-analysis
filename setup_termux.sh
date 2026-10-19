#!/data/data/com.termux/files/usr/bin/bash
# One-time Termux setup for the NLP sentiment project.
# NOT yet verified on a real device: if a step fails, read the message and see README "Troubleshooting".
set -e

pkg update -y
# Heavy scientific packages come prebuilt from Termux (much faster than compiling with pip).
pkg install -y python python-numpy python-scipy matplotlib git unzip clang make pkg-config
# These may or may not exist in your Termux repo; ignore failures and let pip cover them.
pkg install -y python-pandas || true
pkg install -y python-scikit-learn || true

# Do not run "pip install --upgrade pip" in Termux; it can break Termux's pip.
pip install pandas scikit-learn nltk joblib pytest

python -c "import numpy, scipy, pandas, sklearn, nltk, matplotlib, joblib; print('All imports OK')"
python -m nltk.downloader punkt_tab punkt stopwords wordnet averaged_perceptron_tagger_eng averaged_perceptron_tagger
echo "Setup finished. Next: python -m src.train"
