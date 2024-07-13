import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import nest_asyncio

import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', FutureWarning)

from causality_llm.models import direct_gpt
from causality_llm.datasets.download_pushshift import download_pushshift


def save_path_fn(cfg, final=False):
    save_path = cfg.save_path + cfg.data.dataset + "/"
    if final:
        save_path += cfg.date_time + "_"
    save_path += cfg.prompt_type + "_" + cfg.data.dataset + "_" + cfg.model.llm_name + "_" + cfg.experiment_name 
    save_path += '_filtered_posts.csv'
    return save_path

def get_model(cfg):
    prompt_file = open(cfg.prompt_dir + cfg.data.dataset + "_" + cfg.prompt_type + ".txt", "r")
    system_template = prompt_file.read()
    
    post_template = "\n\n## Subreddit \n> This post was found on the subreddit r/{subreddit}."
    post_template += "\n\n## Title \n> This post was titled: {title}"
    post_template += "\n\n## Date Created \n> This post was created on {date_created}."
    post_template += "\n\n## Post \n> {post}"
    post_template += "\n\n> Answer Yes if the post is relevant and No otherwise, and nothing more."
    post_template += "\n\n## Your Answer \n>"

    comment_template = "\n\n## Subreddit \n> This comment was found on the subreddit r/{subreddit}."
    comment_template += "\n\n## Title \n> This comment was in response to a post titled: {title}"
    comment_template += "\n\n## Date Created \n> This comment was created on {date_created}."
    comment_template += "\n\n## Comment \n> {post}"
    comment_template += "\n\n> Answer Yes if the comment is relevant and No otherwise, and nothing more."
    comment_template += "\n## Your Answer \n>"

    key_file = open(cfg.model.key_path, "r") 
    openai_key = key_file.read().rstrip('\n')
    llm = direct_gpt(model_name=cfg.model.llm_name, 
              openai_api_key=openai_key,
              system_template=system_template,
              human_template="",
              temperature=cfg.model.temperature,
              seed=cfg.seed)
    llm.post_template = post_template
    llm.comment_template = comment_template
    return llm

def get_submission_permalink(permalink):
    return '/' + permalink.split('/')[-2] + '/'
    
def get_comment_permalink(permalink):
    return '/' + permalink.split('/')[-3] + '/'

def check(answer):
    return "yes" in answer.lower() and "no" not in answer.lower()

def filter_by_date(df, utc_date_cutoff):
    idx = df["created_utc"].apply(lambda x: int(x) <= utc_date_cutoff)
    return df[idx]

def get_date(utc_timestamp):
    dt = datetime.datetime.fromtimestamp(utc_timestamp, tz=datetime.timezone.utc)
    formatted_date = dt.strftime('%B %d, %Y')
    return formatted_date

def rule_based_filter(post_df, text_field):
    # remove rows where the text field is not of type str
    idx = post_df[text_field].apply(lambda x: isinstance(x, str))
    post_df = post_df.loc[idx]
    idx = post_df["permalink"].apply(lambda x: isinstance(x, str))
    post_df = post_df.loc[idx]
    # remove rows without a score
    post_df = post_df.loc[post_df["score"] != None]
    # remove rows where the submission is deleted or removed
    post_df = post_df.loc[post_df[text_field] != "[deleted]"]
    post_df = post_df.loc[post_df[text_field] != "[removed]"]
    # remove very short comments
    if text_field == "body":
        idx = post_df[text_field].apply(lambda x : len(x.split()) >= 10)
        post_df = post_df.loc[idx]
    # remove posts with "bot" in the author's name
    idx = post_df["author"].apply(lambda x: "bot" not in x.lower())
    post_df = post_df.loc[idx]
    for i, row in post_df.iterrows():
        body = row[text_field]
        # unescape some common html tags
        body = body.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
        body = body.replace('\n', ' ').replace('\t', ' ')
        body = body.strip()
        # drop if there is no space in first 2048 characters
        try:
            tmp = body[: body.rindex(' ', 0, 2048)]
        except ValueError:
            post_df = post_df.drop([i])
            continue
        # drop everything with less than 50% alphabetic characters; space counts
        length_characters = float(len(body))
        filtered = [c for c in body if c.isalpha()]
        if float(len(filtered)) / length_characters < 0.5:
            post_df = post_df.drop([i])
            continue
    return post_df

def check_treatment_mention(lst, treatment_names):
    filtered_lst = []
    for elem in lst:
        matches = [x for x in treatment_names if x in elem["subreddit"].lower()]
        matches += [x for x in treatment_names if x in elem["title"].lower()]
        matches += [x for x in treatment_names if x in elem["post"].lower()]
        if "initial_post" in list(elem.keys()):
            matches += [x for x in treatment_names if x in elem["initial_post"].lower()]
        matches = set(matches)
        if len(matches) > 0:
            elem["treatments"] = list(matches)
            filtered_lst.append(elem)
    return filtered_lst

def check_outcome_mention(lst, outcome_words):
    filtered_lst = []
    for elem in lst:
        matches = [x for x in outcome_words if x in elem["subreddit"].lower()]
        matches += [x for x in outcome_words if x in elem["title"].lower()]
        matches += [x for x in outcome_words if x in elem["post"].lower()]
        if "initial_post" in list(elem.keys()):
            matches += [x for x in outcome_words if x in elem["initial_post"].lower()]
        matches = set(matches)
        if len(matches) > 0:
            elem["outcome_words"] = list(matches)
            filtered_lst.append(elem)
    return filtered_lst

def get_context_post_df(submissions, comments, treatment_names, outcome_words): 
    merged_df = pd.DataFrame(columns=["subreddit", "title", "initial_post", "post", "score", "date_created", "permalink", "treatments", "outcome_words", "author_replies"])
    comments["permalink_processed"] = comments["permalink"].map(lambda x: get_comment_permalink(x))
    for i, submission in submissions.iterrows():
        subreddit = submission["subreddit"]
        title = submission["title"]
        submission_text = submission["selftext"]  
        score = int(submission["score"])
        created_utc = int(submission["created_utc"])
        date_created = get_date(created_utc)
        submission_permalink = get_submission_permalink(submission['permalink'])
        submission_comments = comments[comments['permalink_processed'] == submission_permalink]
        submission_author_comments = submission_comments[submission_comments["author"] == submission["author"]]
        submission_comments = submission_comments.drop(submission_author_comments.index)
        if len(submission_author_comments["body"].to_list()) > 0:
            submission_text += "\n\nThe author also replied with the following in the thread:"
            for reply in submission_author_comments["body"].to_list():
                submission_text += "\n> " + reply
        to_append = [{"subreddit": subreddit,
                      "title": title, 
                      "initial_post": "",
                      "post": submission_text,
                      "score": score,
                      "date_created": date_created,
                      "permalink": submission_permalink,
                      "author_replies": submission_author_comments["body"].to_list()}]
        to_append += [{"subreddit": subreddit,
                       "title": title,
                       "initial_post": submission_text, 
                       "post": str(submission_comments.iloc[j]["body"]),
                       "score": str(submission_comments.iloc[j]["score"]),
                       "date_created": get_date(int(submission_comments.iloc[j]["created_utc"])),
                       "permalink": submission_comments.iloc[j]["permalink"],
                       "author_replies": []} for j in range(len(submission_comments))]
        to_append = check_treatment_mention(to_append, treatment_names)
        to_append = check_outcome_mention(to_append, outcome_words)
        if len(to_append) > 0:
            df_to_append = pd.DataFrame.from_dict(to_append)
            merged_df = pd.concat([merged_df, df_to_append], ignore_index=True)      
    return merged_df

@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    dataset = instantiate(cfg.data.dataset_class)
    try:
        rule_filtered_df = pd.read_csv(save_path_fn(cfg).replace(cfg.model.llm_name, "rule_based"), index_col=0)
    except:
        os.mkdir(cfg.save_path + cfg.data.dataset, exist_ok=True)
        rule_filtered_df = pd.DataFrame()
        for num in range(len(dataset.subreddits)):
            subreddit = dataset.subreddits[num]
            # download submissions and comments from pushshift archives
            download_pushshift(subreddit, "submissions", cfg.save_path + cfg.data.dataset + "/")
            download_pushshift(subreddit, "comments", cfg.save_path + cfg.data.dataset + "/")

            submissions = pd.read_csv(cfg.save_path + cfg.data.dataset + "/" + subreddit + "_submissions.csv")
            comments = pd.read_csv(cfg.save_path + cfg.data.dataset + "/" + subreddit + "_comments.csv")
            if dataset.utc_date_cutoff:
                submissions = filter_by_date(submissions, dataset.utc_date_cutoff)
                comments = filter_by_date(comments, dataset.utc_date_cutoff)
            submissions = rule_based_filter(submissions, "selftext")
            comments = rule_based_filter(comments, "body")
            merged_df = get_context_post_df(submissions, comments, dataset.treatment_names, dataset.outcome_words)
            rule_filtered_df = pd.concat([rule_filtered_df, merged_df], ignore_index=True) 
            rule_filtered_df.to_csv(save_path_fn(cfg).replace(cfg.model.llm_name, "rule_based"))
        rule_filtered_df = rule_filtered_df.drop_duplicates("post")
        rule_filtered_df.to_csv(save_path_fn(cfg).replace(cfg.model.llm_name, "rule_based"))

    llm = get_model(cfg)
    nest_asyncio.apply()
    filtered_posts_df = pd.DataFrame()
    input_dicts, llm_inputs = [], []

    save_path = save_path_fn(cfg)

    if cfg.restore:
        filtered_posts_df = pd.read_csv(save_path, index_col=0)
        rule_filtered_df = rule_filtered_df.iloc[len(filtered_posts_df):]

    for i in tqdm(range(len(rule_filtered_df))):
        row = rule_filtered_df.iloc[i]
        input_dicts.append(row.to_dict())
        row_dict = row[["subreddit", "title", "date_created", "post"]].to_dict()
        if row["initial_post"] == "": # this is a post
            user_input = llm.post_template.format(**row_dict)
        else: # this is a comment
            user_input = llm.comment_template.format(**row_dict)
        llm_inputs.append(user_input)
        if len(llm_inputs) >= cfg.model.batch_size or len(llm_inputs) == len(rule_filtered_df):
            llm_answers = llm.get_outputs(llm_inputs)
            dict_to_save = [{**input_dicts[j], "relevant": llm_answers[j].lower()} for j in range(len(llm_inputs))]
            df_to_save = pd.DataFrame.from_dict(dict_to_save)
            filtered_posts_df = pd.concat([filtered_posts_df, df_to_save], ignore_index=True)
            filtered_posts_df.to_csv(save_path) 
            input_dicts, llm_inputs = [], []
    
    save_path_final = save_path_fn(cfg, final=True)
    filtered_posts_df.to_csv(save_path_final) 

    # keep relevant posts and format for extraction
    relevant_posts = filtered_posts_df.copy()[filtered_posts_df["relevant"] == "yes"]
    post_template = "\n\n## Subreddit \n> This post was found on the subreddit r/{subreddit}."
    post_template += "\n\n## Title \n> This post was titled: {title}"
    post_template += "\n\n## Date Created \n> This post was created on {date_created}."
    post_template += "\n\n## Post \n> {post}"
    post_template += "\n\n## Output \n>"

    comment_template = "\n\n## Subreddit \n> This comment was found on the subreddit r/{subreddit}."
    comment_template += "\n\n## Title \n> This comment was in response to a post titled: {title}"
    comment_template += "\n\n## Date Created \n> This comment was created on {date_created}."
    comment_template += "\n\n## Comment \n> {post}"
    comment_template += "\n\n## Output \n>"
    for i in tqdm(relevant_posts.index.tolist()):
        row = relevant_posts.loc[i]
        row_dict = row[["subreddit", "title", "date_created", "post"]].to_dict()
        if row["initial_post"] == "": # this is a post
            user_input = post_template.format(**row_dict)
        else: # this is a comment
            user_input = comment_template.format(**row_dict)
        relevant_posts.loc[i, "post"] = user_input

    relevant_posts = relevant_posts.drop_duplicates("post")
    relevant_posts.to_csv(save_path.replace(".csv", "_relevant.csv"))
    

if __name__ == "__main__":
    main()