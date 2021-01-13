# -*- coding: utf-8 -*-
import os
import textacy
import logging
from tqdm import tqdm


class TopicModelPermutation:
    def __init__(self, grp_term_matrix, vectorizer, version=None):
        # matrix & model
        self.grp_term_matrix = grp_term_matrix
        self.vectorizer = vectorizer
        self.version = version

        # sklearn.decomposition.<model>
        self.model_types = ["nmf", "lsa", "lda"]

        # cols = number of topics in the model to be initialized
        self.n_topics_list = [2, 3, 4, 5, 6, 7, 8, 9]

        # rows = number of terms
        self.n_terms_list = [10, 30, 50]

    def calc(self, model_dir=None, figure_dir=None, save=True, plot=True):
        logger = logging.getLogger(__name__)
        logger.info("Topic modelling permutation.")

        for model_type in tqdm(self.model_types):
            for n_topics in tqdm(self.n_topics_list):
                for n_terms in tqdm(self.n_terms_list):

                    # init model
                    model = textacy.tm.TopicModel(
                        model=model_type, n_topics=n_topics
                    )

                    # fit model
                    model.fit(self.grp_term_matrix)

                    # transform group-term matrix to group-topic matrix
                    model.transform(self.grp_term_matrix)

                    # save model to disk
                    if save:
                        model.save(
                            os.path.join(
                                model_dir,
                                "BBC_2007_07_04_CORPUS_TEXTACY_{}_TM_{}_{}x{}.{}".format(
                                    self.version,
                                    model_type.upper(),
                                    n_topics,
                                    n_terms,
                                    "pkl",
                                ),
                            )
                        )

                    # termite plot
                    if plot:
                        model.termite_plot(
                            doc_term_matrix=self.grp_term_matrix,
                            id2term=self.vectorizer.id_to_term,
                            topics=-1,
                            n_terms=n_terms,
                            sort_topics_by="index",
                            rank_terms_by="topic_weight",
                            sort_terms_by="seriation",
                            save=os.path.join(
                                figure_dir,
                                "BBC_2007_07_04_CORPUS_TEXTACY_{}_TM_{}_{}x{}.{}".format(
                                    self.version,
                                    model_type.upper(),
                                    n_topics,
                                    n_terms,
                                    "png",
                                ),
                            ),
                            rc_params={"dpi": 300},
                        )
