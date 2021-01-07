# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Online sources:
#   https://towardsdatascience.com/building-a-topic-modeling-pipeline-with-spacy-and-gensim-c5dc03ffc619
# -----------------------------------------------------------------------------
import os
import glob
import click
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

# load spacy lm
nlp = spacy.load("en_core_web_lg")


def lemmatizer(doc):
    # This takes in a doc of tokens from the NER and lemmatizes them.
    # Pronouns (like "I" and "you" get lemmatized to '-PRON-', so I'm removing those.
    doc = [token.lemma_ for token in doc if token.lemma_ != "-PRON-"]
    doc = u" ".join(doc)
    return nlp.make_doc(doc)


def remove_stopwords(doc):
    # This will remove stopwords and punctuation.
    # Use token.text to return strings, which we'll need for Gensim.
    doc = [
        token.text
        for token in doc
        if token.is_stop != True and token.is_punct != True
    ]
    return doc


def init(stop_word_list=[]):

    # set max length (found a text with len=13'406'690)
    nlp.max_length = 30000000

    # Updates spaCy's default stop words list with additional words
    nlp.Defaults.stop_words.update(stop_word_list)

    # Iterates over the words in the stop words list and resets the "is_stop" flag
    for word in STOP_WORDS:
        lexeme = nlp.vocab[word]
        lexeme.is_stop = True

    # The add_pipe function appends our functions to the default pipeline.
    nlp.add_pipe(lemmatizer, name="lemmatizer", after="ner")
    nlp.add_pipe(remove_stopwords, name="stopwords", last=True)


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath, preprocess=True):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # initialise spacy nlp model
    init()

    # compile list of documents (slower, but more robust than os.listdir)
    dp_string = os.path.join(input_filepath, "BBC_2007_07_04_TXT", "*.txt")
    file_list = glob.glob(dp_string)
    file_list = sorted(file_list)

    # initialise corpus and fill
    corpus = {}
    for i, file_path in enumerate(tqdm(file_list)):

        # extract metadata of file name
        fname = file_path.split("/")[-1].split(".")[0].split("_")
        if len(fname) == 2:
            river_basin, year = fname
            month = np.nan
        elif len(fname) == 3:
            river_basin, year, month = fname
        else:
            pass

        # read txt file
        with open(file_path) as f_input:
            # read txt file
            text = f_input.read()

            # split conditional on pre-processing
            if preprocess:
                fid = "_PROCESSED"

                # df cols
                cols = [
                    "basin",
                    "year",
                    "month",
                    "text_original",
                    "text_processed",
                ]

                # optional: pre-process with spacy
                # memory limit: "The v2.x parser and NER models require roughly
                # 1GB of temporary memory per 100,000 characters in the input."
                text_pr = nlp(text, disable=["ner", "parser"])

                # append to corpus
                corpus[i] = [river_basin, year, month, text, text_pr]
            else:
                fid = ""

                # df cols
                cols = ["basin", "year", "month", "text_original"]

                # append to corpus
                corpus[i] = [river_basin, year, month, text]

    # convert corpus from dict to df
    df_corpus = pd.DataFrame.from_dict(corpus, orient="index", columns=cols)

    # re-index
    dfm_corpus = df_corpus.set_index(["basin", "year"])

    # count chars per row
    dfm_corpus["n_chars_processed"] = dfm_corpus["text_processed"].apply(
        lambda x: len(x)
    )

    # save
    fpath_out = os.path.join(
        output_filepath, "BBC_2007_07_04_CORPUS{}_INDEXED.csv".format(fid)
    )
    df_corpus.to_csv(fpath_out)

    fpath_out = os.path.join(
        output_filepath, "BBC_2007_07_04_CORPUS{}_INDEXED.pkl".format(fid)
    )
    df_corpus.to_pickle(fpath_out)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # sub-directories
    data_raw = os.path.join(project_dir, "data", "raw")
    data_interim = os.path.join(project_dir, "data", "interim")
    data_processed = os.path.join(project_dir, "data", "processed")

    main(data_raw, data_interim, preprocess=True)
