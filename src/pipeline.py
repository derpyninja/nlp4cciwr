# -*- coding: utf-8 -*-
import os
import logging
import en_core_web_lg
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from references import nlp_dicts
from src.data import make_datasets


# -----------------------------------------------------------------------------
# general configuration
# -----------------------------------------------------------------------------

# version of run
version = "V5"

# corpus name
fname_corpus = "BBC_2007_07_04_CORPUS_TEXTACY_{}.bin.gz".format(version)

# init logger
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

# not used in this stub but often useful for finding various files
project_dir = Path(__file__).resolve().parents[1]

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())

# sub-directories
model_dir = os.path.join(project_dir, "models")
data_raw = os.path.join(project_dir, "data", "raw")
data_interim = os.path.join(project_dir, "data", "interim")
data_processed = os.path.join(project_dir, "data", "processed")
figure_dir = os.path.join(project_dir, "reports", "figures")


# -----------------------------------------------------------------------------
# load and configure spaCy nlp pipeline
# -----------------------------------------------------------------------------
nlp = en_core_web_lg.load()
nlp.max_length = int(30 * 1e6)
nlp.remove_pipe("parser")
nlp.remove_pipe("ner")

# -----------------------------------------------------------------------------
# 1) create corpus
# -----------------------------------------------------------------------------
corpus = make_datasets.create_corpus(
    input_filepath=os.path.join(data_raw, "BBC_2007_07_04_TXT_V2", "*.txt"),
    output_filepath=os.path.join(data_processed, fname_corpus),
    nlp=nlp,
    specific_stopwords=nlp_dicts.stopwords_bbc_monitoring,
    return_data=True,
)

# some stats
print(corpus)
print(corpus.n_docs, corpus.n_sents, corpus.n_tokens)

# -----------------------------------------------------------------------------
# 2) descriptive statistics
# -----------------------------------------------------------------------------
