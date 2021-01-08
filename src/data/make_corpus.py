# -*- coding: utf-8 -*-
import os
import re
import glob
import textacy
import logging
import numpy as np
from tqdm import tqdm
import en_core_web_lg
from pathlib import Path
from textacy import preprocessing
from dotenv import find_dotenv, load_dotenv


def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).

    Parameters
    ----------
    input_filepath: str
        Path to unmutable raw data (BBC Monitoring txt files)
    output_filepath: str
        Path where corpus is saved

    Returns
    -------
    corpus: textacy.Corpus
        Corpus created from BBC Monitoring data stored in binary format
        and zipped for optimal compression
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating corpus from raw BBC Monitoring data")

    # -------------------------------------------------------------------------
    # load and configure spacy nlp model
    # -------------------------------------------------------------------------
    nlp = en_core_web_lg.load()
    nlp.max_length = 30000000
    nlp.Defaults.stop_words |= {
        "hyperlink",
        "HYPERLINK",
    }
    # -------------------------------------------------------------------------
    # compile list of documents (slower, but more robust than os.listdir)
    # -------------------------------------------------------------------------
    dp_string = os.path.join(input_filepath, "BBC_2007_07_04_TXT", "*.txt")
    file_list = glob.glob(dp_string)
    file_list = sorted(file_list)
    # -------------------------------------------------------------------------
    # read in text stream
    # -------------------------------------------------------------------------
    records = []
    for file_path in tqdm(file_list):
        # ---------------------------------------------------------------------
        # extract metadata of file name
        # ---------------------------------------------------------------------
        fname = file_path.split("/")[-1].split(".")[0].split("_")
        if len(fname) == 2:
            river_basin, year = fname
            month = np.nan
        elif len(fname) == 3:
            river_basin, year, month = fname
        else:
            pass
        # ---------------------------------------------------------------------
        # read and pre-process with nlp pipeline
        # ---------------------------------------------------------------------
        with open(file_path) as f_input:

            # read text file
            text = f_input.read()

            # 1) replace
            text = preprocessing.replace_currency_symbols(text, replace_with="")
            text = preprocessing.replace_emails(text, replace_with="")
            text = preprocessing.replace_emojis(text, replace_with="")
            text = preprocessing.replace_hashtags(text, replace_with="")
            text = preprocessing.replace_numbers(text, replace_with="")
            text = preprocessing.replace_phone_numbers(text, replace_with="")
            text = preprocessing.replace_urls(text, replace_with="")
            text = preprocessing.replace_user_handles(text, replace_with="")

            # 2) remove
            text = preprocessing.remove_accents(text)
            text = preprocessing.remove_punctuation(text)
            text = re.sub("[^A-Za-z0-9]+", " ", text)

            # 3) normalise
            text = preprocessing.normalize_hyphenated_words(text)
            text = preprocessing.normalize_quotation_marks(text)
            # text = preprocessing.normalize_repeating_chars(text)
            text = preprocessing.normalize_whitespace(text)

            # 4) create doc with metadata
            metadata = {"basin": river_basin, "year": year, "month": month}
            doc = textacy.make_spacy_doc((text, metadata), lang=nlp)

            # 5) append record
            records.append(doc)

    # ---------------------------------------------------------------------
    # build corpus (can be re-loaded via textacy.Corpus.load)
    # ---------------------------------------------------------------------
    corpus = textacy.Corpus(nlp, data=records)
    output_fname = "BBC_2007_07_04_CORPUS_TEXTACY.bin.gz"
    corpus.save(os.path.join(output_filepath, output_fname))
    return None


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
    main(data_raw, data_processed)
