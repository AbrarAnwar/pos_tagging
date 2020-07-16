# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np

vocab = {}
tagCounts = {}
transitionCounts = {}
observationCounts = {}
capitalCounts = {}
hyphenCounts = {}

# let this N be the frequency required to become an unknown word
# anything less than N is viewed as an unknown word
N = 10
# this lis comes from a blog from esllibrary.com plus stuff I thought up
noun_suffixes = ['ion', 'acy', 'age', 'ance', 'ence', 'hood', 'ar', 'or', 'ism', 'ist', 'ment', 'ness', 'ty', 'ship', 'er', 'dom']
adjective_suffixes = ['able', 'al', 'ant', 'ed', 'ent', 'ful', 'ible', 'ic', 'ing', 'ive', 'less', 'ous', 'y', 'th']
adverb_suffixes = ['ly']
verb_suffixes = ['ate', 'en', 'ify', 'ise', 'ize', 'ing']


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # let's create our vocab list first and have counts as well
    with open(train_file) as train_sentences:
        for line in train_sentences:
            for wordAndTag in line.split():
                idx = wordAndTag.rfind('/')
                currTag = wordAndTag[idx+1:]
                currWord = wordAndTag[:idx]

                vocab.setdefault(currWord, 0)
                vocab[currWord] += 1
    sizeLeftOut = 0
    for key, value in vocab.items():
        if value < N:
            sizeLeftOut += 1

    with open(train_file) as train_sentences:
        for line in train_sentences:

            # add these start words into it
            prevWord = '<s>'
            prevTag = '<s>'

            tagCounts.setdefault(prevTag, 0)
            tagCounts[prevTag]+=1

            for wordAndTag in line.split():
                idx = wordAndTag.rfind('/')
                currTag = wordAndTag[idx+1:]
                currWord = wordAndTag[:idx]
                if vocab[currWord] < N:
                    unkW = '<UNK>'
                    if '-' in currWord:
                        unkW = '<UNK-HYP>'
                    elif any(c.isupper() for c in currWord):
                        unkW = '<UNK-CAP>'
                    elif any(currWord.endswith(s) for s in adverb_suffixes):
                        unkW = '<UNK-ADV>'
                    elif any(currWord.endswith(s) for s in noun_suffixes):
                        unkW = '<UNK-N>'
                    elif any(currWord.endswith(s) for s in verb_suffixes):
                        unkW = '<UNK-VRB>'
                    elif any(currWord.endswith(s) for s in adjective_suffixes):
                        unkW = '<UNK-ADJ>'
                    elif any(c.isdigit() for c in currWord):
                        unkW = '<UNK-NUM>'


                    observationCounts.setdefault((unkW, currTag), 0)
                    observationCounts[(unkW, currTag)] += 1
                    tagCounts.setdefault(currTag, 0)
                    tagCounts[currTag] += 1
                    #currWord = unkW

                # the follow create a dict entry if it is not already in the dict
                tagCounts.setdefault(currTag, 0)
                transitionCounts.setdefault((prevTag, currTag), 0)
                observationCounts.setdefault((currWord, currTag), 0)

                # now we add one to all of them
                tagCounts[currTag] += 1
                transitionCounts[(prevTag, currTag)] += 1
                observationCounts[(currWord, currTag)] += 1

                # we add the special case to help model unknown words
                if '-' in currWord:
                    hyphenCounts.setdefault(currTag, 0)
                    hyphenCounts[currTag] += 1
                if any(c.isupper() for c in currWord):
                    capitalCounts.setdefault(currTag, 0)
                    capitalCounts[currTag] += 1

                # set the prevs 
                prevWord = currWord
                prevTag = currTag
            
            # deal with the end of a sentence. most of it is for consistency really
            tagCounts.setdefault('</s>', 0)
            transitionCounts.setdefault((prevTag, '</s>'), 0)
            observationCounts.setdefault(('</s>', '</s>'), 0)

            tagCounts['</s>'] += 1
            transitionCounts[(prevTag, '</s>')] += 1
            observationCounts[('</s>', '</s>')] += 1

    with open(model_file, "w") as f:
        f.write(str(len(tagCounts)) + '\n')
        f.write(str(len(transitionCounts)) + '\n')
        f.write(str(len(observationCounts)) + '\n')
        f.write(str(len(hyphenCounts)) + '\n')
        f.write(str(len(capitalCounts)) + '\n')
        writeDict(tagCounts, f)
        writeDictTuples(transitionCounts, f)
        writeDictTuples(observationCounts, f)
        writeDict(hyphenCounts, f)
        writeDict(capitalCounts, f)

    print('Finished...')

# given a dict d and a file f, writes d to the file
def writeDict(d, f):
    for key, val in d.items():
        f.write(str(key) + ' ' + str(val)+'\n')
    
def writeDictTuples(d, f):
    for key, val in d.items():
        f.write(key[0] + ' ' + key[1] + ' ' + str(val)+'\n')

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
