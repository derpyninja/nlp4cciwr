# -*- coding: utf-8 -*-
import logging
import textacy
import pandas as pd


def read_corpus(fpath, language_model, store_user_data=True):
    """

    Parameters
    ----------
    fpath : str
        file path
    language_model : spaCy
         nlp
    store_user_data : bool
        custom extension attributes

    Returns
    -------
    textacy.Corpus
        corpus instance
    """
    logger = logging.getLogger(__name__)
    logger.info("Reading pre-computed corpus.")

    return textacy.Corpus.load(
        lang=language_model, filepath=fpath, store_user_data=store_user_data
    )


def read_group_term_matrix(fpath, kind="csr"):
    # read group-term matrix
    return textacy.io.matrix.read_sparse_matrix(filepath=fpath, kind=kind)


def read_vectorizer(fpath):
    # read pre-trained vectorizer
    return pd.read_pickle(fpath)
