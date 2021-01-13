# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt


def word_counts(corpus, data_dir=None, figure_dir=None, version=None, n=30):
    logger = logging.getLogger(__name__)
    logger.info("Visualising word counts.")

    # calc
    word_counts_wcount = corpus.word_counts(
        weighting="count", as_strings=True, normalize=None
    )
    word_counts_wfreq = corpus.word_counts(
        weighting="freq", as_strings=True, normalize=None
    )

    # df
    df_word_counts_wcount = pd.DataFrame.from_dict(
        data=word_counts_wcount, orient="index", columns=["count"]
    )
    df_word_counts_wfreq = pd.DataFrame.from_dict(
        data=word_counts_wfreq, orient="index", columns=["freq"]
    )

    # merge
    if (df_word_counts_wfreq.index == df_word_counts_wcount.index).all():
        df_word_counts = pd.concat(
            [df_word_counts_wcount, df_word_counts_wfreq], axis=1
        )

    # sanity check
    assert (
        (
            df_word_counts.sort_values("count", ascending=False)
            == df_word_counts.sort_values("freq", ascending=False)
        )
        .all()
        .all()
    )

    # sort
    df_word_counts = df_word_counts.sort_values("count", ascending=False)

    # save
    if data_dir is not None:
        df_word_counts.to_pickle(
            os.path.join(
                data_dir,
                "BBC_2007_07_04_CORPUS_TEXTACY_{}_WORDCOUNT.pkl".format(
                    version
                ),
            )
        )

    # plot
    ax = df_word_counts.head(n).plot.bar(
        sharex=True, subplots=True, color="grey", legend=False
    )
    plt.tight_layout()

    # save fig
    if figure_dir is not None:
        plt.savefig(
            os.path.join(
                figure_dir,
                "BBC_2007_07_04_CORPUS_TEXTACY_{}_WORDCOUNT_N{}.png".format(
                    version, n
                ),
            ),
            dpi=300,
            bbox_inches="tight",
        )

    return ax, df_word_counts


def word_document_counts(
    corpus, data_dir=None, figure_dir=None, version=None, n=30
):
    logger = logging.getLogger(__name__)
    logger.info("Visualising word-document counts.")

    # calc
    word_doc_counts_wfreq = corpus.word_doc_counts(
        weighting="freq", as_strings=True
    )
    word_doc_counts_wcount = corpus.word_doc_counts(
        weighting="count", as_strings=True
    )
    word_doc_counts_widf = corpus.word_doc_counts(
        weighting="idf", as_strings=True
    )

    # df
    df_word_doc_counts_wfreq = pd.DataFrame.from_dict(
        data=word_doc_counts_wfreq, orient="index", columns=["freq"]
    )
    df_word_doc_counts_wcount = pd.DataFrame.from_dict(
        data=word_doc_counts_wcount, orient="index", columns=["count"]
    )
    df_word_doc_counts_widf = pd.DataFrame.from_dict(
        data=word_doc_counts_widf, orient="index", columns=["idf"]
    )

    # merge
    if (
        df_word_doc_counts_wfreq.index == df_word_doc_counts_wcount.index
    ).all():
        df_word_doc_counts = pd.concat(
            [
                df_word_doc_counts_wcount,
                df_word_doc_counts_wfreq,
                df_word_doc_counts_widf,
            ],
            axis=1,
        )

    # sanity check
    assert (
        (
            df_word_doc_counts.sort_values("count", ascending=False)
            == df_word_doc_counts.sort_values("freq", ascending=False)
        )
        .all()
        .all()
    )

    # sort
    df_word_doc_counts = df_word_doc_counts.sort_values(
        "count", ascending=False
    )

    if data_dir is not None:
        df_word_doc_counts.to_pickle(
            os.path.join(
                data_dir,
                "BBC_2007_07_04_CORPUS_TEXTACY_{}_WORDDOCCOUNT.pkl".format(
                    version
                ),
            )
        )

    # plot
    ax = df_word_doc_counts.head(n).plot.bar(
        sharex=True, subplots=True, color="grey", legend=False
    )
    plt.tight_layout()

    # save fig
    if figure_dir is not None:
        plt.savefig(
            os.path.join(
                figure_dir,
                "BBC_2007_07_04_CORPUS_TEXTACY_{}_WORDDOCCOUNT_N{}.png".format(
                    version, n
                ),
            ),
            dpi=150,
            bbox_inches="tight",
        )

    return ax, df_word_doc_counts
