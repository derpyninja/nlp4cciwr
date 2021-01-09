# -*- coding: utf-8 -*-
import os
import re
import glob
import logging
import textacy
import gensim
import numpy as np
from tqdm import tqdm
import en_core_web_lg
from pathlib import Path
from textacy import preprocessing
from dotenv import find_dotenv, load_dotenv
from references import nlp_dicts


def preprocess_text(
    text, char_count_filter=True, stopwords=None, min_len=2, max_len=15
):
    """
    Pre-processing steps prior to spaCy nlp pipeline. Optional filtering of
    tokens based on character length.

    Parameters
    ----------
    text : str
    char_count_filter : bool
    stopwords : iterable, None
    min_len : int
    max_len : int

    Returns
    -------
    text : str
        pre-processed text
    """
    # 1) convert to lower case for robust stop-word recognition
    text = text.lower()

    # 2) normalise
    text = preprocessing.normalize_quotation_marks(text)
    # text = preprocessing.normalize_repeating_chars(text)
    text = preprocessing.normalize_hyphenated_words(text)
    text = preprocessing.normalize_whitespace(text)

    # 3) replace
    text = preprocessing.replace_currency_symbols(text)
    text = preprocessing.replace_emails(text)
    text = preprocessing.replace_emojis(text)
    text = preprocessing.replace_hashtags(text)
    text = preprocessing.replace_numbers(text)
    text = preprocessing.replace_phone_numbers(text)
    text = preprocessing.replace_urls(text)
    text = preprocessing.replace_user_handles(text)

    # 4) remove
    text = preprocessing.remove_accents(text)
    text = preprocessing.remove_punctuation(text)
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # keep text and numbers

    # 5) optionally remove tokens based on length
    if char_count_filter & (stopwords is not None):
        # filter based on token length
        tokens = gensim.utils.simple_preprocess(
            doc=text, min_len=min_len, max_len=max_len
        )
        # filter case-specific words
        tokens = [token for token in tokens if token not in stopwords]

        # convert processed list of tokens back to one string
        text = " ".join(tokens)
    else:
        raise NotImplementedError("Not implemented.")

    return text


def main(
    input_filepath, output_filepath, case_specific_stopwords=None, save=True
):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).

    Parameters
    ----------
    input_filepath: str
        Folder path storing un-mutable raw data. Use a wildcard within the
         file name to filter files via glob.glob.
    output_filepath: str
        File path where corpus should be saved.
    case_specific_stopwords: iterable, None
        Case specific stopwords that are worth deleting before going into the
        spaCy pipeline to prevent memory allocation problems.
    save: bool
        True if corpus should be saved, else otherwise.

    Returns
    -------
    corpus: textacy.Corpus
        Corpus created from BBC Monitoring data stored in binary format
        and zipped for optimal compression
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating corpus from raw BBC Monitoring data")

    # load and configure spacy nlp model
    # https://stackoverflow.com/questions/52557058/spacy-nlp-pipeline-order-of-operations
    # -------------------------------------------------------------------------
    nlp = en_core_web_lg.load()
    nlp.max_length = int(30 * 1e6)
    nlp.remove_pipe("parser")
    nlp.remove_pipe("ner")

    # compile list of documents (slower, but more robust than os.listdir)
    # -------------------------------------------------------------------------
    file_list = glob.glob(input_filepath)
    file_list = sorted(file_list)

    # iteratively read in text stream
    # -------------------------------------------------------------------------
    records = []
    for file_path in tqdm(file_list):

        # extract metadata of file name
        # ---------------------------------------------------------------------
        fname = file_path.split("/")[-1].split(".")[0].split("_")
        if len(fname) == 2:
            river_basin, year = fname
            month = np.nan
        elif len(fname) == 3:
            river_basin, year, month = fname
        else:
            raise NotImplementedError("Check needed!")

        metadata = {"basin": river_basin, "year": year, "month": month}

        # read and pre-process with nlp pipeline
        # ---------------------------------------------------------------------
        with open(file_path) as f_input:
            # 1) read raw text file
            text_raw = f_input.read()

            # 2) pre-process with utils (textacy only, or textacy & gensim)
            text = preprocess_text(
                text_raw,
                char_count_filter=True,
                stopwords=case_specific_stopwords,
                min_len=3,
                max_len=15,
            )

            # 3) create doc with metadata
            doc = textacy.make_spacy_doc(data=(text, metadata), lang=nlp)

            # 4) append record
            records.append(doc)

    # build corpus
    # ---------------------------------------------------------------------
    corpus = textacy.Corpus(nlp, data=records)
    if save:
        corpus.save(output_filepath)
    else:
        return corpus


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

    # run
    main(
        input_filepath=os.path.join(data_raw, "BBC_2007_07_04_TXT", "*.txt"),
        output_filepath=os.path.join(
            data_processed, "BBC_2007_07_04_CORPUS_TEXTACY_V2.bin.gz"
        ),
        case_specific_stopwords=nlp_dicts.stopwords_bbc_monitoring,
        save=True,
    )
