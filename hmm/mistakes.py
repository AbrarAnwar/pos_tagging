# python3.5 eval.py <output_file_absolute_path> <reference_file_absolute_path>
# make no changes in this file

import os
import sys


if __name__ == "__main__":
    out_file = sys.argv[1]
    reader = open(out_file)
    out_lines = reader.readlines()
    reader.close()
    ref_file = sys.argv[2]
    reader = open(ref_file)
    ref_lines = reader.readlines()
    reader.close()

    model_file = sys.argv[3]

    tagCounts = {}
    transitionCounts = {}
    observationCounts = {}

    # load the model data
    with open(model_file) as f:
        tagN = int(f.readline())
        transN = int(f.readline())
        obsN = int(f.readline())
        x = f.readline()
        x = f.readline()

        for i in range(tagN):
            line = f.readline()
            line = line.split()
            tagCounts.setdefault(line[0], int(line[1]))
        for i in range(transN):
            line = f.readline()
            line = line.split()
            transitionCounts.setdefault((line[0], line[1]), int(line[2]))
        for i in range(obsN):
            line = f.readline()
            line = line.split()
            observationCounts.setdefault((line[0], line[1]), int(line[2]))



    if len(out_lines) != len(ref_lines):
        print('Error: No. of lines in output file and reference file do not match.')
        exit(0)

    vocab = set()
    for key in observationCounts.keys():
        (word, tag) = key
        vocab.add(word)

    total_tags = 0
    matched_tags = 0
    OOVCount = 0
    OOVRight = 0
    totalWrong = 0
    isReasonable = 0
    notReasonable = 0
    for i in range(0, len(out_lines)):
        cur_out_line = out_lines[i].strip()
        cur_out_tags = cur_out_line.split(' ')
        cur_ref_line = ref_lines[i].strip()
        cur_ref_tags = cur_ref_line.split(' ')
        total_tags += len(cur_ref_tags)
        for j in range(0, len(cur_ref_tags)):
            if cur_out_tags[j] == cur_ref_tags[j]:
                matched_tags += 1
                idx = cur_out_tags[j].rfind('/')
                outTag = cur_out_tags[j][idx+1:]
                outWord = cur_out_tags[j][:idx]
                trueTag = cur_ref_tags[j][idx+1:]
                trueWord = cur_ref_tags[j][:idx]
                if trueWord not in vocab:
                    #print('GOT OOV RIGHT')
                    OOVRight += 1

                #print('correct {}'.format(cur_out_tags[j]))
            else:
                idx = cur_out_tags[j].rfind('/')
                outTag = cur_out_tags[j][idx+1:]
                outWord = cur_out_tags[j][:idx]
                trueTag = cur_ref_tags[j][idx+1:]
                trueWord = cur_ref_tags[j][:idx]
                predCount = observationCounts.get((outWord, outTag))
                trueCount = observationCounts.get((trueWord, trueTag))
                if predCount > trueCount:
                    isReasonable += 1
                else:
                    notReasonable += 1
                if not('-' in trueWord or any(c.isupper() for c in trueWord)) and trueWord not in vocab:
                    print('pred: {} true: {}'.format(cur_out_tags[j], cur_ref_tags[j]))
                    print('\tcount for pred: {}'.format(predCount))
                    print('\tcount for true: {}'.format(trueCount))
                    print('\tisReasonable: {}'.format(predCount>trueCount))

                    print('is any digit: {}'.format(any(c.isdigit() for c in trueWord)))
                
                if trueWord not in vocab:
                    print('\tWORD NOT IN VOCAB')
                    OOVCount += 1
                totalWrong += 1
    print('isReasonable, notReasonable {} {}'.format(isReasonable, notReasonable))
    print('total wrong:', totalWrong)
    print('total OOV wrong', OOVCount)
    print('total OOV right', OOVRight)
    print(matched_tags, total_tags)
    print("Accuracy=", float(matched_tags) / total_tags)
