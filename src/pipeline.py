# -*- coding: utf-8 -*-
import os
import logging
import en_core_web_lg
import seaborn as sns
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# custom module components
from references import nlp_dicts
from src.data import io, make_corpus
from src.features import extract
from src.models import train_model, predict_model
from src.visualization import visualize

# visualisation settings
sns.set_context("poster")
sns.set(rc={"figure.figsize": (16, 9.0)})
sns.set_style("ticks")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# version of run
version = "V6"

# file names
fname_corpus = "BBC_2007_07_04_CORPUS_TEXTACY_{}.bin.gz".format(version)
fname_gt_matrix = (
    "BBC_2007_07_04_CORPUS_TEXTACY_{}_GROUPTERMMATRIX_STEP1.npz".format(version)
)
fname_vectorizer = "BBC_2007_07_04_CORPUS_TEXTACY_{}_VECTORIZER.pkl".format(
    version
)

# load and configure spaCy nlp pipeline
nlp = en_core_web_lg.load()
nlp.max_length = int(30 * 1e6)
nlp.remove_pipe("parser")
nlp.remove_pipe("ner")

# -----------------------------------------------------------------------------
# Initialisation
# -----------------------------------------------------------------------------

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

# file paths
fpath_corpus = os.path.join(data_processed, fname_corpus)
fpath_vectorizer = os.path.join(model_dir, fname_vectorizer)
fpath_gt_matrix = os.path.join(data_processed, fname_gt_matrix)

# -----------------------------------------------------------------------------
# 1) IO/Corpus
# -----------------------------------------------------------------------------
if not os.path.exists(fpath_corpus):
    corpus = make_corpus.create_corpus(
        input_filepath=os.path.join(data_raw, "BBC_2007_07_04_TXT_V2", "*.txt"),
        output_filepath=os.path.join(data_processed, fname_corpus),
        nlp=nlp,
        specific_stopwords=nlp_dicts.stopwords_bbc_monitoring,
        return_data=True,
    )
else:
    corpus = io.read_corpus(
        fpath=fpath_corpus, language_model=nlp, store_user_data=True
    )

# -----------------------------------------------------------------------------
# 2) Feature Extraction
# -----------------------------------------------------------------------------
if not os.path.exists(fpath_gt_matrix):
    vectorizer = train_model.group_vectorizer()

    tokenized_docs, basin_group, year_group = extract.tokenize_corpus(
        corpus=corpus
    )

    grp_term_matrix = train_model.group_vectorizer_fit_transform(
        vectorizer=vectorizer,
        tokenized_docs=tokenized_docs,
        group_data=basin_group,
        data_dir=data_processed,
        model_dir=model_dir,
        version=version,
        save=True,
    )
else:
    vectorizer = io.read_vectorizer(fpath=fpath_vectorizer)
    grp_term_matrix = io.read_group_term_matrix(fpath=fpath_gt_matrix)

# -----------------------------------------------------------------------------
# 3) Topic Modelling
# -----------------------------------------------------------------------------
compute_topic_models = False

if compute_topic_models:
    tm_permutation = predict_model.TopicModelPermutation(
        grp_term_matrix=grp_term_matrix, vectorizer=vectorizer, version=version
    )

    tm_permutation.calc(
        model_dir=model_dir, figure_dir=figure_dir, save=True, plot=True
    )

# -----------------------------------------------------------------------------
# 4) Visualise
# -----------------------------------------------------------------------------
visualize.word_counts(
    corpus=corpus,
    data_dir=data_processed,
    figure_dir=figure_dir,
    version=version,
)
visualize.word_document_counts(
    corpus=corpus,
    data_dir=data_processed,
    figure_dir=figure_dir,
    version=version,
)
