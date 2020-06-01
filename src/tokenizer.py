import wordsegment as ws
from nltk.tokenize import TweetTokenizer
import csv
import regex as re
import sys
import pickle
"""
Utils for tokenizing and datacleaning
"""

"""
Tweet tokenizing script based on the one provided by pennington GloVe coupled with NLTK tweettokenizer
Date: 4th October 2019
"""

ws.load()

FLAGS = re.MULTILINE | re.DOTALL


def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(
            ["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS)
        )
    return result


def segmnt(text): return " ".join(ws.segment(text.group(0)))


def remove_hashtag(text): return re.sub(r'\#([a-zA-Z0-9_]+)', segmnt, text)


def allcaps(text):
    text = text.group()
    return text.lower()


def penning_tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "")
    text = re_sub(r"@\w+", "")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "")
    text = re_sub(r"{}{}p+".format(eyes, nose), "")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "")
    text = re_sub(r"/", " / ")
    text = re_sub(r"<3", "")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "")
    text = re_sub(r"#\S+", "")
    text = re_sub(r"([!?.]){2,}", r"\1 ")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 ")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)
    text = text.strip()

    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)

    return text.lower()


def penning_preprocess(data):
    #     tk = TweetTokenizer()
    return [penning_tokenize(sent) for sent in data]




if __name__ == '__main__':
    text = "I TEST alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
    print(text)
    print(penning_tokenize(text))
