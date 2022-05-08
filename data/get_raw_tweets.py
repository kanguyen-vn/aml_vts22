import os
import sys
import logging
import json
import time
import traceback
from dotenv import load_dotenv
import datetime
from datetime import datetime as dt, timedelta, timezone
import random
import tweepy
import pandas as pd
from json.decoder import JSONDecodeError
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))


def try_mkdir(end_dir):
    try:
        os.mkdir(end_dir)
    except FileExistsError:
        logger.warning(f"{end_dir} already created.")
    return end_dir


raw_tweets_dir = try_mkdir(os.path.join(current_dir, "raw_tweets"))
metadata_file_name = "pagination_metadata.json"
start_date_all = "2010-01-01"
max_requests_per_15min = 300

includes_types = ["tweets", "users", "places", "media", "polls"]
tweet_fields = ["id", "created_at", "text", "author_id", "conversation_id", "in_reply_to_user_id", "referenced_tweets",
                "public_metrics", "entities", "context_annotations", "geo", "attachments", "source", "possibly_sensitive",
                "reply_settings", "lang", "withheld"]
user_fields = ["id", "username", "name", "created_at", "description", "entities", "location", "verified", "public_metrics",
               "pinned_tweet_id", "profile_image_url", "protected", "url", "withheld"]
media_fields = ["media_key", "type", "public_metrics",
                "preview_image_url", "width", "height", "alt_text", "duration_ms", "url"]
place_fields = ["id", "name", "full_name", "contained_within",
                "country", "country_code", "place_type", "geo"]
poll_fields = ["id", "duration_minutes",
               "end_datetime", "options", "voting_status"]
includes_fields = {k: v for k, v in zip(includes_types, [
                                        tweet_fields, user_fields, place_fields, media_fields, poll_fields])}


def build_query(filename="query.txt"):
    with open(os.path.join(current_dir, filename)) as f:
        queries = f.read().splitlines()
    modifier = " lang:en -is:retweet"
    return "(" + queries[0] + ")" + modifier


def get_paginator(
    client,
    query,
    start_time=None,
    since_id=None,
    max_results=100,
    end_time=None,
    until_id=None,
    pagination_token=None,
    minimal=False
):
    """Get Tweepy Paginator to get all tweets with query."""
    if start_time is None and since_id is None:
        start_time = start_date_all
    if end_time is not None and not end_time.endswith("Z"):
        end_time = f"{end_time}T00:00:00Z"
    if start_time is not None and not start_time.endswith("Z"):
        start_time = f"{start_time}T00:00:00Z"
    kwargs = dict(
        query=query,
        start_time=start_time,
        end_time=end_time,
        since_id=since_id,
        until_id=until_id,
        max_results=max_results,
        pagination_token=pagination_token
    )
    try:
        if minimal:
            return tweepy.Paginator(client.search_all_tweets, **{k: v for k, v in kwargs.items() if v is not None})
        return tweepy.Paginator(client.search_all_tweets,
                                **{k: v for k, v in kwargs.items() if v is not None},
                                expansions=["author_id", "referenced_tweets.id", "in_reply_to_user_id", "geo.place_id",
                                            "entities.mentions.username", "referenced_tweets.id.author_id",
                                            "attachments.poll_ids", "attachments.media_keys"],
                                media_fields=media_fields,
                                place_fields=place_fields,
                                poll_fields=poll_fields,
                                tweet_fields=tweet_fields,
                                user_fields=user_fields
                                )
    except Exception as e:
        if isinstance(e, tweepy.TweepyException):
            logger.error(f"API error when getting paginator: {e}")
    return None


def load_pagination_metadata(parent_dir, meta_file_name=metadata_file_name):
    """Load tokens and iteration count from metadata."""
    full_path = os.path.join(parent_dir, meta_file_name)
    if not os.path.exists(full_path):
        # logger.error(f"Metadata file does not exist: {full_path}.")
        return None, None, 0
    try:
        with open(full_path) as f:
            metadata = json.load(f)
        previous_token = metadata["previous_token"] if "previous_token" in metadata else ""
        next_token = metadata["next_token"] if "next_token" in metadata else ""
        iteration = metadata["iteration"] if "iteration" in metadata else 0
        return previous_token, next_token, iteration
    except JSONDecodeError:
        logger.error(f"Metadata file is empty: {full_path}.")
    return None, None, 0


def sleep(seconds=60 * 11):
    """Sleep 11 minutes when 15-min rate limit hits (grabbing data already takes 5+ mins)."""
    logger.info(f"Sleeping for {seconds / 60} minutes...")
    # Print remaining time on console
    for i in range(seconds, 0, -1):
        print(
            f"Time remaining: {datetime.timedelta(seconds=i)}",
            end="\r",
            flush=True
        )
        time.sleep(1)


def find_most_recent_chunk(date_dir):
    """Get most recent chunk number out of the available data files."""
    dir_name = os.path.basename(os.path.normpath(date_dir))
    files = [file for file in os.listdir(date_dir)
             if os.path.isfile(os.path.join(date_dir, file))
             and file != metadata_file_name]
    if len(files) == 0:
        return 0
    # File name: f"{dir_name}_{chunk}.csv"
    files = [os.path.splitext(file[len(dir_name) + 1:])[0]
             for file in files]
    files = [int(file) for file in files if file.isnumeric()]
    return max(files)


def get_path_and_df(date_dir, chunk, includes_type=None):
    """Get path and dataframe from full dir path and chunk number."""
    columns = tweet_fields if includes_type is None else includes_fields[includes_type]
    dirname = os.path.basename(os.path.normpath(date_dir))
    csv_path = os.path.join(
        date_dir, f"{dirname}_{includes_type + '_' if includes_type is not None else ''}{str(chunk).rjust(2, '0')}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, header=0)
    else:
        df = pd.DataFrame(columns=columns)
    return csv_path, df


# def get_conversations_path_and_df(data_dir, includes_type=None):
#     """Get path and dataframe from full dir path and chunk number."""
#     columns = tweet_fields if includes_type is None else includes_fields[includes_type]
#     dirname = os.path.basename(os.path.normpath(data_dir))
#     csv_path = os.path.join(
#         data_dir, f"{dirname}_conversations{'_' + includes_type if includes_type is not None else ''}.csv")
#     if os.path.exists(csv_path):
#         df = pd.read_csv(csv_path, header=0)
#     else:
#         df = pd.DataFrame(columns=columns)
#     return csv_path, df


def append_to_df_and_reset_list(data):
    """data[0] is the list, data[1] is the df. For efficient appending."""
    data[1] = pd.concat(
        [data[1], pd.DataFrame.from_records(data[0])], ignore_index=True)
    # data[1] = data[1].append(data[0], ignore_index=True)
    return [[], data[1]]


def sample_tweets(client, count_goal=100000000, start_date="2012-04-01", end_date="2022-04-01"):
    """Sample tweets randomly over a period of time."""
    word_count = 0
    iteration = 0
    sample_tweets_dir = os.path.join(current_dir, "samples")
    try_mkdir(sample_tweets_dir)

    def get_sample_df(chunk):
        path = os.path.join(sample_tweets_dir,
                            f"sample_{str(chunk).rjust(2, '0')}.csv")
        return path, pd.DataFrame()

    def timestamp(date):
        dt_object = dt.strptime(date, r"%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(round(dt_object.timestamp()))

    def to_string(timestamp):
        return dt.utcfromtimestamp(timestamp).strftime(r"%Y-%m-%dT%H:%M:%SZ")

    max_results = 100

    query = "lang:en -is:retweet the -the"
    sampled_timestamps = []

    random.seed(27)

    start_timestamp = timestamp(start_date)
    end_timestamp = timestamp(end_date)

    chunk = 0
    data_csv_path, df = get_sample_df(chunk)
    main_data = [[], df]

    while word_count < count_goal:
        start_second = random.randrange(start_timestamp, end_timestamp)
        while start_second in sampled_timestamps:
            start_second = random.randrange(start_timestamp, end_timestamp)
        start, end = to_string(start_second), to_string(start_second + 1)
        logger.info(f"Start second: {start}")
        paginator = get_paginator(
            client, query, start_time=start, end_time=end, max_results=max_results, minimal=True)
        if paginator is None:
            return
        page_iter = iter(paginator)

        finished = False

        while not finished:
            try:
                # Get next page of results
                page = next(page_iter)
                iteration += 1

                if page.data is None:
                    logger.warning("No data from page.")
                    finished = True
                    time.sleep(1)
                    continue

                if iteration % 10 == 0:
                    logger.info(
                        f"Saving data from page, request {iteration}/{max_requests_per_15min}.")

                count = 0
                for tweet in page.data:
                    main_data[0].append(tweet.data)
                    count += len(tweet.data["text"].split())

                logger.info(f"Got {count} words this second.")
                word_count += count

                # Save all to file
                main_data = append_to_df_and_reset_list(main_data)
                main_data[1].to_csv(data_csv_path, index=False)

                if not finished and iteration == max_requests_per_15min:
                    logger.info(f"Finished chunk: {data_csv_path}.")
                    chunk += 1  # Next chunk
                    data_csv_path, df = get_sample_df(chunk)  # New dataframes
                    main_data = [[], df]
                    iteration = 0  # Back to beginning
                    logger.info(
                        "End of chunk; rate limit hit (no exception). Saving progress and taking a break...")
                    sleep()
                time.sleep(1)  # 1 request per second
            except Exception as e:
                if isinstance(e, tweepy.TooManyRequests):
                    logger.error("Rate limit hit. Taking a break...")
                    sleep()
                    paginator = get_paginator(
                        client, query, start_time=start, end_time=end, max_results=max_results, minimal=True)
                    if not paginator:
                        return
                    page_iter = iter(paginator)
                elif isinstance(e, StopIteration):
                    finished = True
                else:
                    logger.error(f"Error: {e}")
                    logger.info("Shutting down...")
                    traceback.print_exc()
                    sys.exit(1)
        main_data = append_to_df_and_reset_list(main_data)
        main_data[1].to_csv(data_csv_path, index=False)

        logger.info(f"Word count: {word_count}/{count_goal}.")

        # logger.info(f"Finished all: {data_csv_path}.")
        # sleep()


def get_all_tweets(client, query):
    """Get all tweets with query."""
    # # Folder name by start of month starting from {start_date_all} to the present
    # names_by_month = pd.date_range(
    #     start_date_all, dt.now(), freq="MS").strftime(r"%Y-%m-%d").tolist()
    # # Full path to folder
    # names_by_month_dirs = [try_mkdir(os.path.join(raw_tweets_dir, name))
    #                        for name in names_by_month]

    # names_by_year = ["2022-03-01"]

    # Folder name by start of month starting from {start_date_all} to the present
    names_by_year = pd.date_range(
        start_date_all, dt.now(), freq="YS").strftime(r"%Y-%m-%d").tolist()

    # Full path to folder
    names_by_year_dirs = [try_mkdir(os.path.join(raw_tweets_dir, name))
                          for name in names_by_year]

    # Number of results to be returned by client.search_all_tweets
    max_results = 100

    for index, start_date in enumerate(names_by_year):
        date_dir = names_by_year_dirs[index]
        # Tokens for pagination; iteration to make sure each chunk contains 300 requests for consistency
        previous_token, next_token, iteration = load_pagination_metadata(
            date_dir)
        # If already at the end (next_token == "")
        if next_token and len(next_token) == 0:
            logger.warning(
                f"Already grabbed data from {start_date} to {names_by_year[index + 1]}.")
            continue

        if index == len(names_by_year) - 1:
            # end_date = "2022-04-01"
            os.rmdir(date_dir)
            return  # Wait until month has passed to get tweets
        else:
            end_date = names_by_year[index + 1]

        logger.info("Getting paginator...")
        paginator = get_paginator(client, query, start_time=start_date, end_time=end_date,
                                  max_results=max_results, pagination_token=next_token)
        if paginator is None:
            return

        finished = False
        # Find latest chunk => keep appending to it
        chunk = find_most_recent_chunk(date_dir)
        if iteration == max_requests_per_15min:  # End of chunk
            chunk += 1
            iteration = 0

        data_csv_path, df = get_path_and_df(date_dir, chunk)
        main_data = [[], df]
        includes_csv_paths = {}
        includes_dfs = {}
        for includes_type in includes_types:
            includes_csv_path, includes_df = get_path_and_df(
                date_dir, chunk, includes_type)
            includes_csv_paths[includes_type] = includes_csv_path
            includes_dfs[includes_type] = [[], includes_df]

        logger.info("- Getting iterator for paginator...")
        page_iter = iter(paginator)
        while not finished:
            try:
                # Get next page of results
                page = next(page_iter)
                iteration += 1

                if page.data is None:
                    logger.warning("No data from page.")
                    finished = True
                    time.sleep(1)
                    continue

                logger.info(
                    f"Saving data from page, request {iteration}/{max_requests_per_15min}.")

                # Save information in page.includes
                for includes_type in includes_types:
                    if includes_type in page.includes:
                        for item in page.includes[includes_type]:
                            row = item if isinstance(item, dict) else item.data
                            # includes_dfs[includes_type] = includes_dfs[includes_type].append(
                            #     row, ignore_index=True)
                            includes_dfs[includes_type][0].append(row)

                for tweet in page.data:
                    # df = df.append(tweet.data, ignore_index=True)
                    main_data[0].append(tweet.data)

                # Save all to file
                main_data = append_to_df_and_reset_list(main_data)
                main_data[1].to_csv(data_csv_path, index=False)
                for includes_type in includes_dfs:
                    includes_dfs[includes_type] = append_to_df_and_reset_list(
                        includes_dfs[includes_type])
                    includes_dfs[includes_type][1].to_csv(
                        includes_csv_paths[includes_type], index=False)

                with open(os.path.join(date_dir, metadata_file_name), "w") as f:
                    # Store metadata
                    previous_token = next_token
                    if "next_token" not in page.meta:
                        finished = True
                        page.meta["next_token"] = ""
                    else:
                        next_token = page.meta["next_token"]
                    if previous_token:
                        page.meta["previous_token"] = previous_token
                    page.meta["iteration"] = iteration
                    json.dump(page.meta, f)

                # End of chunk (30,000 tweets)
                if not finished and iteration == max_requests_per_15min:
                    logger.info(f"Finished chunk: {data_csv_path}.")
                    chunk += 1  # Next chunk
                    data_csv_path, df = get_path_and_df(
                        date_dir, chunk)  # New dataframes
                    main_data = [[], df]
                    for includes_type in includes_types:
                        includes_csv_path, includes_df = get_path_and_df(
                            date_dir, chunk, includes_type)
                        includes_csv_paths[includes_type] = includes_csv_path
                        includes_dfs[includes_type] = [[], includes_df]
                    iteration = 0  # Back to beginning
                    logger.info(
                        "End of chunk; rate limit hit (no exception). Saving progress and taking a break...")
                    sleep()
                time.sleep(1)  # 1 request per second
            except Exception as e:
                if isinstance(e, tweepy.TooManyRequests):
                    logger.error("Rate limit hit. Taking a break...")
                    sleep()
                    logger.info("Resetting iterator...")
                    previous_token, next_token, iteration = load_pagination_metadata(
                        date_dir)  # Same iteration since haven't incremented when exception would've been thrown
                    if next_token == "":
                        # Only if next(page_iter) is called right after a page is done processing
                        next_token = None
                    paginator = get_paginator(client, query, start_time=start_date,
                                              end_time=end_date, max_results=max_results, pagination_token=next_token)
                    if not paginator:
                        return
                    page_iter = iter(paginator)
                elif isinstance(e, StopIteration):
                    finished = True
                else:
                    logger.error(f"Error: {e}")
                    logger.info("Shutting down...")
                    traceback.print_exc()
                    sys.exit(1)
        main_data = append_to_df_and_reset_list(main_data)
        main_data[1].to_csv(data_csv_path, index=False)
        for includes_type in includes_dfs:
            includes_dfs[includes_type] = append_to_df_and_reset_list(
                includes_dfs[includes_type])
            includes_dfs[includes_type][1].to_csv(
                includes_csv_paths[includes_type], index=False)
        logger.info(f"Finished all: {data_csv_path}.")
        sleep()


def print_tweet_counts():
    raw_tweets_path = os.path.join(current_dir, "raw_tweets")
    total = 0
    tally = []
    for dirpath, _, filenames in os.walk(raw_tweets_path):
        dirname = os.path.basename(os.path.normpath(dirpath))
        if not dirname[0].isnumeric():
            continue
        names = [filename for filename in filenames if os.path.splitext(filename)[
            1] == ".csv"]
        names = [name for name in names if name[len(
            dirname) + 1:-4].isnumeric()]
        paths = [os.path.join(dirpath, name) for name in names]
        count_by_dir = sum(
            len(pd.read_csv(path, header=0, dtype=str)) for path in paths)
        print(f"{dirname}: {count_by_dir}")
        total += count_by_dir
        tally.append(count_by_dir)
    print("Total:", total)

    tally.append(total)
    print(" & ".join([f"{number:,}" for number in tally]))

    xs = range(2010, 2022)
    tally = tally[:-2]
    plt.plot(xs, tally)
    plt.xlabel("Year")
    plt.ylabel("Number of tweets")
    plt.xticks(xs)
    plt.tight_layout()
    plt.show()


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

    # Initialize environment variables
    load_dotenv()
    bearer_token = os.getenv("OAUTH_BEARER_TOKEN")

    # Initialize Tweepy
    client = tweepy.Client(bearer_token=bearer_token)

    # Get query
    # start_time = time.time()
    # query = build_query()

    # if query is not None and len(query) > 0:
    #     get_all_tweets(client, query)

    # end_time = time.time()

    # logger.info(
    #     f"Finished getting data. Total time: {timedelta(seconds=end_time - start_time)}.")

    start_time = time.time()

    sample_tweets(client)

    end_time = time.time()

    logger.info(
        f"Finished sampling tweets. Total time: {timedelta(seconds=end_time - start_time)}.")
