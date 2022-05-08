import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import torch

# BERTopic
from bertopic._bertopic import BERTopic

# Dimension reduction
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Embeddings
from flair.embeddings import TransformerDocumentEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer

# Evaluation
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaModel


def train_bertopic(data, embedding_model, dimension_reduction_model):
    vectorizer_model = CountVectorizer(ngram_range=(1, 1), min_df=1)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=5,
        top_n_words=20,
        min_topic_size=30,
        verbose=True,
        low_memory=True,
        vectorizer_model=vectorizer_model,
        umap_model=dimension_reduction_model,
    )
    topics, _ = topic_model.fit_transform(data.tolist())
    return topic_model, topics


def get_coherence_score(data, topic_model, topics, coherence):
    # Extract vectorizer and tokenizer from BERTopic
    vectorizer = topic_model.vectorizer_model
    tokenizer = vectorizer.build_tokenizer()

    # Extract features for Topic Coherence evaluation
    tokens = [tokenizer(doc) for doc in data]
    # tokens = [token for token in tokens if token!='']

    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [
        [words for words, _ in topic_model.get_topic(topic) if words != ""]
        for topic in range(len(set(topics)) - 1)
    ]

    # Evaluate
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokens,
        corpus=corpus,
        dictionary=dictionary,
        coherence=coherence,
    )
    return coherence_model.get_coherence()


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "preprocessed_tweets",
        "all_tweets.csv",
    )
    df = pd.read_csv(data_path, header=0)
    df = df.text.dropna()[:5000]

    # # Train LDA

    # Train BERTopic using BERTweet base vs. BERT base

    bertweet = TransformerDocumentEmbeddings("vinai/bertweet-base")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding_models = [bertweet, sentence_model]
    embedding_model_names = ["BERTweet", "BERT base"]

    tsne = TSNE(n_components=5, init="pca", method="exact", random_state=2022)
    # print(isinstance(tsne, TSNE))
    # pca = PCA(n_components=5, random_state=2022)
    # umap = UMAP(n_neighbors=35, n_components=5, min_dist=0.0, metric="euclidean", random_state=2022)

    # dimension_reduction_models = [tsne, pca, umap]

    for model, name in zip(embedding_models, embedding_model_names):
        # t-SNE model
        # tsne = TSNE(n_components=5, init="pca", method="exact", random_state=2022)
        topic_model, topics = train_bertopic(df, model, tsne)  # pca)  # umap)  # tsne)
        print(topics)
        score = get_coherence_score(df, topic_model, topics, "u_mass")
        print(
            f"{datetime.now().strftime('%H:%M:%S')}: {name} with t-SNE UMass score: {score}."
        )

    # bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    # print(vars(bertweet))

    # print(str(type(bertweet)))
    # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    # print(str(type(tokenizer)))

    # line = "This is an example sentence"

    # input_ids = torch.tensor([tokenizer.encode(line)])

    # with torch.no_grad():
    #     features = bertweet(input_ids)

    # output = features.pooler_output.numpy()
    # print(type(output))
    # print(output.shape)

    # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    # sentences = ["This is an example sentence", "Each sentence is converted"]

    # embeddings = sentence_model.encode(sentences)
    # # print(embeddings)
    # print(type(embeddings[0]))
    # print(len(embeddings[0]))

    # Train BERTopic using BERTweet
