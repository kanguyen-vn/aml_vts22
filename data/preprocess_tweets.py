import os
import sys
import re
from ast import literal_eval
import logging
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import preprocessor as prep

logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
tk = TweetTokenizer(preserve_case=False, reduce_len=True)
raw_tweets_dir = os.path.join(current_dir, "raw_tweets")


def try_mkdir(dirpath):
    try:
        os.mkdir(dirpath)
    except FileExistsError:
        pass


remove_all_stopwords = True
if remove_all_stopwords:
    preprocessed_tweets_dir_name = "preprocessed_tweets_all_stopwords_removed"
    with open(os.path.join(current_dir, "stopwords_all.txt")) as f:
        stops = f.read().splitlines()
else:
    preprocessed_tweets_dir_name = "preprocessed_tweets"
    with open(os.path.join(current_dir, "stopwords.txt")) as f:
        stops = f.read().splitlines()

preprocessed_tweets_dir = os.path.join(
    current_dir, preprocessed_tweets_dir_name)
try_mkdir(preprocessed_tweets_dir)


def preprocess(tweet):  # , hashtags=None):
    """Preprocess a tweet."""
    processed = tweet

    # Segment hashtags
    # if hashtags is not None:
    #     hashtags.sort(key=lambda d: d["end"], reverse=True)
    #     for hashtag in hashtags:
    #         # processed = processed.replace(hashtag, seg_tw.segment(hashtag))
    #         processed = processed[:hashtag["start"] + 1] + \
    #             seg_tw.segment(hashtag["tag"]) + processed[hashtag["end"]:]

    # Remove hashtag symbols
    processed = re.sub(r"#", "", processed)

    # Change whitespaces to spaces
    # processed = " ".join(processed.split())

    # Preprocess using tweet-preprocessor
    prep.set_options(prep.OPT.URL, prep.OPT.MENTION, prep.OPT.RESERVED,
                     prep.OPT.EMOJI, prep.OPT.SMILEY, prep.OPT.NUMBER)
    processed = prep.clean(processed)

    # Tokenize
    tokens = tk.tokenize(processed)

    # Remove single-letter and two-letter strings
    tokens = [token for token in tokens if len(token) > 2]

    # Remove stopwords, punctuation, and numbers
    tokens = [token for token in tokens if (token not in stops) and (not all(
        c in punctuation or c.isspace() for c in token)) and (not token.isnumeric())]

    # tokens = [token for token in tokens if not any(
    #     c in punctuation for c in token)]

    # Lemmatizing
    # tagged = pos_tag(tokens)
    # for index, pair in enumerate(tagged):
    #     token, tag = pair
    #     pos = get_wordnet_pos(tag)
    #     if token in ["us", "isis"]:
    #         continue
    #     lemmatized = lemmatizer.lemmatize(
    #         token, pos=pos) if pos else lemmatizer.lemmatize(token)
    #     # Try verb if plural noun doesn't work (e.g. "a stack of books" vs "books a hotel room")
    #     if tag == "NNS" and lemmatized == token:
    #         lemmatized = lemmatizer.lemmatize(
    #             token, pos=VERB)
    #     tokens[index] = lemmatized

    return " ".join(tokens)


def split_sequence_of_hashtags(tweet):
    """Split many hashtags strung together. e.g. 'abc #Hashtag1#Hashtag2' => 'abc #Hashtag1 #Hashtag2'"""
    tweet = re.sub(r"#{2,}", "#", tweet)
    i = len(tweet) - 1
    while i > 0:
        if tweet[i] == "#" and tweet[i - 1] != " ":
            tweet = tweet[:i] + " " + tweet[i:]
        i -= 1

    # hashtags = []
    # for i in range(len(tweet) - 1):
    #     if tweet[i] == "#" and tweet[i + 1] != " ":
    #         next_space = tweet.find(" ", i + 1)
    #         if next_space == -1:
    #             next_space = len(tweet)
    #         hashtags.append({"start": i, "end": next_space,
    #                          "tag": tweet[i + 1:next_space]})
    return tweet  # , hashtags


def concat_all(data_dir=preprocessed_tweets_dir, start_time=None, end_time=None):
    """Concatenate all chunks into one big dataframe."""
    logger.info("Concatenating files...")
    all_tweets_path = os.path.join(data_dir, "all_tweets.csv")
    df = pd.DataFrame()
    # query_dirnames = [dirname for dirname in os.listdir(
    #     data_dir) if os.path.isdir(os.path.join(data_dir, dirname))]
    # for query_dirname in query_dirnames:
    #     query_dirpath = os.path.join(data_dir, query_dirname)
    # Make sure concatenation in correct order => sort YYYY-MM-DD descending
    dirnames = sorted([name for name in os.listdir(data_dir) if os.path.isdir(
        os.path.join(data_dir, name))], reverse=True)
    for dirname in dirnames:
        month_dirpath = os.path.join(data_dir, dirname)
        filenames = sorted([filename for filename in os.listdir(month_dirpath) if os.path.splitext(
            filename)[1] == ".csv" and len(os.path.splitext(filename)[0]) == 13])
        for filename in filenames:
            logger.info(f"- Adding {filename}...")
            filepath = os.path.join(month_dirpath, filename)
            df = pd.concat(
                [df, pd.read_csv(filepath, header=0)], ignore_index=True)
    # if len(query_dirnames) > 1:
    #     df.sort_values(by="id", ascending=False,
    #                    inplace=True, ignore_index=True)
    start_time = df.created_at.min(
    ) if start_time is None else f"{start_time}T00:00:00Z"
    end_time = df.created_at.max(
    ) if end_time is None else f"{end_time}T00:00:00Z"
    df = df[(pd.to_datetime(df.created_at) >= pd.Timestamp(start_time)) & (pd.to_datetime(
        df.created_at) < pd.Timestamp(end_time))].reset_index(drop=True)
    assert df.created_at.is_monotonic_decreasing, "Concatenation in incorrect order"

    logger.info("Saving...")
    df.to_csv(all_tweets_path, index=False)
    logger.info("Concatenation complete.")
    return df


def preprocess_all(raw_tweets_dir=raw_tweets_dir, preprocessed_tweets_dir=preprocessed_tweets_dir):
    """Preprocess all chunks in raw tweets folder and output data in new folder."""
    for dirpath, dirnames, filenames in os.walk(raw_tweets_dir):
        for dirname in dirnames:
            if dirpath == raw_tweets_dir:
                try_mkdir(os.path.join(preprocessed_tweets_dir, dirname))
            else:
                try_mkdir(os.path.join(preprocessed_tweets_dir,
                                       os.path.relpath(dirpath, raw_tweets_dir), dirname))
        for filename in filenames:
            split = os.path.splitext(filename)
            if split[1] != ".csv":
                continue
            if len(split[0]) > 13:  # only process main tweet files: YYYY-MM-DD_xx.csv
                continue
            logger.info(f"Preprocessing {filename}...")
            raw_chunk_path = os.path.join(dirpath, filename)
            preprocessed_chunk_path = os.path.join(
                preprocessed_tweets_dir, os.path.relpath(dirpath, raw_tweets_dir), filename)
            df = pd.read_csv(raw_chunk_path, header=0)
            new_df_rows = []
            for i in range(len(df)):
                if i % 1000 == 0 and i > 0:
                    logger.info(f"- Done preprocessing {i}/{len(df)} items.")
                tweet = df.text.iat[i]
                ex = None
                try:
                    entities = literal_eval(df.entities.iat[i])
                except Exception as e:
                    ex = e
                # if ex is not None or "hashtags" not in entities:  # string of hashtags
                #     tweet, hashtags = split_sequence_of_hashtags(tweet)
                # else:
                #     hashtags = entities["hashtags"]
                # preprocessed_tweet = preprocess(tweet, hashtags)
                # row = {"id": df.id.iat[i], "preprocessed_tweet": preprocessed_tweet, "hashtags": [
                #     hashtag["tag"] for hashtag in hashtags]}
                tweet = split_sequence_of_hashtags(tweet)
                preprocessed_tweet = preprocess(tweet)
                row = {
                    "id": df.id.iat[i], "created_at": df.created_at.iat[i], "text": preprocessed_tweet}
                new_df_rows.append(row)
            new_df = pd.DataFrame(new_df_rows)
            new_df.to_csv(preprocessed_chunk_path, index=False)
            logger.info(f"- Done preprocessing {len(df)}/{len(df)} items.")


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(
        os.path.join(current_dir, "get_raw_tweets.log"))
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s [%(levelname)s] %(message)s")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger = logging.getLogger(__name__)

    # preprocess_all()
    concat_all(raw_tweets_dir, start_time="2012-04-01")
