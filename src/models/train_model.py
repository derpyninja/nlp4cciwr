# -*- coding: utf-8 -*-
import os
import pickle
import textacy
import logging


def group_vectorizer(
    tf_type="linear",
    apply_idf=True,
    idf_type="standard",
    apply_dl=False,
    dl_type="linear",
    norm="l2",
    min_df=0.3,
    max_df=0.95,
    max_n_terms=None,
    vocabulary_terms=None,
    vocabulary_grps=None,
):

    return textacy.vsm.GroupVectorizer(
        tf_type=tf_type,  # {"linear", "sqrt", "log", "binary"}
        apply_idf=apply_idf,
        idf_type=idf_type,  # {"standard", "smooth", "bm25"}
        apply_dl=apply_dl,
        dl_type=dl_type,  # {"linear", "sqrt", "log"}
        norm=norm,  # {"l1", "l2"} or None
        min_df=min_df,  # Filter terms whose document frequency is less than ``min_df``
        max_df=max_df,  # Filter terms whose document frequency is greater than ``max_df``,
        max_n_terms=max_n_terms,
        vocabulary_terms=vocabulary_terms,
        vocabulary_grps=vocabulary_grps,
    )


def group_vectorizer_fit_transform(
    vectorizer,
    tokenized_docs,
    group_data,
    data_dir=None,
    model_dir=None,
    version=None,
    save=True,
):
    logger = logging.getLogger(__name__)
    logger.info("Computing group-term matrix.")

    # compute group-term matrix
    grp_term_matrix = vectorizer.fit_transform(tokenized_docs, group_data)

    if save:
        # save group-term matrix to disk as a single .npz file (numpy binary format)
        textacy.io.matrix.write_sparse_matrix(
            data=grp_term_matrix,
            filepath=os.path.join(
                data_dir,
                "BBC_2007_07_04_CORPUS_TEXTACY_{}_GROUPTERMMATRIX_STEP1".format(
                    version
                ),
            ),
            compressed=True,
        )

        # save fitted vectorizer
        pickle.dump(
            vectorizer,
            open(
                os.path.join(
                    model_dir,
                    "BBC_2007_07_04_CORPUS_TEXTACY_{}_VECTORIZER.pkl".format(
                        version
                    ),
                ),
                "wb",
            ),
        )

    return grp_term_matrix
