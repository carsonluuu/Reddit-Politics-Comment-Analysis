#!/usr/bin/env python

"""Clean comment text for easier parsing."""

from __future__ import print_function

import re
import string
import argparse
import json
import os.path
import sys
from bz2 import BZ2File

__author__ = "Jiahui Lu"
__email__ = "carsonluuu@gmail.com"

# Some useful data.
_CONTRACTIONS = {
    "tis": "'tis",
    "aint": "ain't",
    "amnt": "amn't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hell": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "ol": "'ol",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "somethings": "something's",
    "thatll": "that'll",
    "thats": "that's",
    "thatd": "that'd",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "wed've",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "whod've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}

# You may need to write regular expressions.

def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings
    1. The parsed text.
    2. The unigrams
    3. The bigrams
    4. The trigrams
    """

    # YOUR CODE GOES BELOW:

    bounding_punctuation = [".", "!", ":", ",", ";", "?"]

    context = text
    #context = "I'm afraid I can't explain myself, sir. Because I am not myself, you see?"
    #context = peek[0]
    context = re.sub(r"\t\n", " ", context)
    context = re.sub(r"http\S+", "", context)
    context = re.sub(r"\s{2,}", " ", context)
    context = re.findall(r"[\w'/\-%$]+|[.,!?;:]", context)

    context = ' '.join(context).lower()

    context = context.replace("0 , 0", "0,0")
    context = context.replace("i . e", "i.e")
    context = context.replace("e . g", "e.g")

    ############parsed_text############
    parsed_text = context

    words = context.split()

    ############unigrams############
    unigram = []
    i = 0
    while i < len(words):
        if words[i] not in bounding_punctuation:
            unigram.append(words[i])
        i += 1
    unigrams = ' '.join(unigram)

    ############bigrams############
    bigram = []
    i = 1
    while i < len(words):
        if words[i - 1] not in bounding_punctuation \
        and words[i] not in bounding_punctuation:
            bigram.append(words[i - 1] + "_" + words[i])
        i += 1
    bigrams = ' '.join(bigram)
    #print(bigrams)

    ############trigram############
    trigram = []
    i = 2
    while i < len(words):
        if words[i - 2] not in bounding_punctuation\
        and words[i - 1] not in bounding_punctuation\
        and words[i] not in bounding_punctuation:
            trigram.append(words[i - 2] + "_" + words[i - 1] + "_" + words[i])
        i += 1
    trigrams = ' '.join(trigram)
    #print(trigrams)

    return [parsed_text, unigrams, bigrams, trigrams]


if __name__ == "__main__":
    # This is the Python main function.
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.

    # YOUR CODE GOES BELOW.

    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    extention = os.path.splitext(args.file)[1]

    if (extention == ".json"):
        reddit_file = open(args.file, "r")
    elif (extention == ".bz2"):
        reddit_file = BZ2File(args.file)
    else:
        print("Please use file with right type (either .json or .bz2)")
        sys.exit()

#    cnt = 0
    ans = []
    for line in reddit_file:
        reddit = json.loads(line.strip())
        comment = reddit['body']
        ans.append(sanitize(comment))
#        if (cnt > 10):
#            break
#        cnt += 1
    for idx in ans:
       print(idx)

    reddit_file.close()
