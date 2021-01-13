# -*- coding: utf-8 -*-
import textacy
import textacy.vsm


def tokenize_corpus(
    corpus,
    ngrams=(1, 2),
    entities=False,
    normalize="lemma",
    as_strings=True,
    filter_stops=True,
    filter_nums=True,
    include_pos={"ADJ", "NOUN", "VERB"},
    min_freq=2,
):
    tokenized_docs, basin_group, year_group = textacy.io.unzip(
        (
            doc._.to_terms_list(
                ngrams=ngrams,
                entities=entities,
                normalize=normalize,
                as_strings=as_strings,
                filter_stops=filter_stops,
                filter_nums=filter_nums,
                include_pos=include_pos,
                min_freq=min_freq,
            ),
            doc._.meta["basin"],
            doc._.meta["year"],
        )
        for doc in corpus
    )

    return tokenized_docs, basin_group, year_group
