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

    Parameters
    ----------
    input_filepath
    output_filepath

    Returns
    -------

    """
    logger = logging.getLogger(__name__)
    logger.info("Extracting features from corpus.")
    pass
