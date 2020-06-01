import csv
import wordsegment as ws
ws.load()
from nltk.tokenize import TweetTokenizer #https://www.nltk.org/api/nltk.tokenize.html
from nltk.corpus import stopwords
import re

def interweave(doc,labels):
    """
        Interweaves the negative and positive examples
        The documents with more examples should be primary
        doc[0]: should be the majority sample
        doc[1]: should be the minority sample
    """
    primary = doc[0]
    secondary = doc[1]

    primary_label = labels[0]
    secondary_label = labels[1]
    
    weave = []
    weave_labels = []
    for i in range(len(primary)):
        weave.append(primary[i])
        weave_labels.append(primary_label)
        try:
            weave.append(secondary[i])
            weave_labels.append(secondary_label)
        except:
            r = random.randint(0,len(secondary)-1)
            weave.append(secondary[r])
            weave_labels.append(secondary_label)
    
    
    return (weave, weave_labels)


def openHasocFile(file_location):
    file = []
    data = []
    label = []
    max_len = 0
    with open(file_location,'r') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        count = 0
        for row in tsvin:
            file.append(row)
            data.append(row[1])
            label.append(row[2])

    #first entry is just the row name, hence removed
    data.pop(0)
    label.pop(0)
    label = [ 0 if i == 'NOT' else 1 for i in label]
    return data,label


def find_max_len(data):
    tknz = TweetTokenizer()
    max_len = 0
    for sentence in data:
        l = len(tknz.tokenize(sentence))
        if  l > max_len:
            max_len = l
    return max_len
       
def preprocess(data):
    """
    Preprocessing Unit as described in Davidson et. al.
    """

    def remove_url(text): return re.sub(r'http\S+', '', text)

    def remove_sym(text): return re.sub(r'\&#\d*;{1}', '', text)

    def remove_amp(text): return re.sub(r'\&[a-zA-Z0-9]+;', '', text)

    def remove_rt(text): return re.sub(r'RT', '', text)

    def remove_redundant(text): return re.sub(
        r'[!\.\?\:\'\"]{2,}', lambda x: ' '+x.group(0)[0]+' ', text)

    def remove_spaces(text): return re.sub(r'[\s]{2,}', ' ', text)

    def handle(text): return text.group(
        0)[1:-1] if text.group(0)[-1] == ':' else text.group(0)[1:]

    def proc_handles(text): return re.sub(
        r'@([A-Za-z0-9_]+)[\:]*', handle, text)

    def remove_slash(text): return re.sub(r"[a-zA-Z]+\\\'[a-zA-Z]+", "'", text)

    def segmnt(text): return " ".join(ws.segment(text.group(0)))

    def remove_hashtag(text): return re.sub(r'\#([a-zA-Z0-9_]+)', segmnt, text)

    data = [remove_hashtag(sent) for sent in data] #not working
    data = [remove_url(sent) for sent in data]
    data = [remove_sym(sent) for sent in data]
    data = [remove_amp(sent) for sent in data]
    data = [remove_rt(sent) for sent in data]
    data = [remove_redundant(sent) for sent in data]
    data = [remove_spaces(sent) for sent in data]
    data = [remove_slash(sent) for sent in data]
    data = [proc_handles(sent) for sent in data]  # not working. FIX

    return data