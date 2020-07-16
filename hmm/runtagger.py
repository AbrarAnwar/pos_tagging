# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np


noun_suffixes = ['ion', 'acy', 'age', 'ance', 'ence', 'hood', 'ar', 'or', 'ism', 'ist', 'ment', 'ness', 'ty', 'ship', 'er', 'dom']
adjective_suffixes = ['able', 'al', 'ant', 'ed', 'ent', 'ful', 'ible', 'ic', 'ing', 'ive', 'less', 'ous', 'y', 'th']
adverb_suffixes = ['ly']
verb_suffixes = ['ate', 'en', 'ify', 'ise', 'ize', 'ing']
const = .0001
discount = .0001

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.

    # set the basic variables
    tagCounts = {}
    transitionCounts = {}
    observationCounts = {}
    hyphenCounts = {}
    capitalCounts = {}

    # load the model data
    with open(model_file) as f:
        
        tagN = int(f.readline())
        transN = int(f.readline())
        obsN = int(f.readline())
        hyphN = int(f.readline())
        capN = int(f.readline())

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
        for i in range(hyphN):
            line = f.readline()
            line = line.split()
            hyphenCounts.setdefault(line[0], int(line[1]))
        for i in range(capN):
            line = f.readline()
            line = line.split()
            capitalCounts.setdefault(line[0], int(line[1]))

    # create a tag list so we can maintain order of tags for our matricies
    tagList = []
    for tag in tagCounts.keys():
        if tag != '<s>' and tag != '</s>':
            tagList.append(tag)
    
    # create the transition matrix for the viterbi algorithm
    np.set_printoptions(threshold=sys.maxsize)
    A = createTransitionMatrix(transitionCounts, tagCounts, tagList)
    
    vocab = set()
    for key in observationCounts.keys():
        (word, tag) = key
        vocab.add(word)

    unkValsGivenTag = {}
    for tag in tagList:
        unkTotalCountGivenTag = 0
        unkSeenGivenTag = 0
        allSeenGivenTag = 0
        for (w, t), value in observationCounts.items():
            if t == tag and '<UNK' in w:
                unkTotalCountGivenTag += value
                unkSeenGivenTag += 1
            if t == tag:
                allSeenGivenTag += 1
        unkValsGivenTag[tag] = (unkTotalCountGivenTag, unkSeenGivenTag, allSeenGivenTag)



    # start reading in test data
    with open(test_file) as f:
        with open(out_file, "w") as out:
            for line in f:
                sentenceList = line.split()
                B = createEmissionMatrix(observationCounts, tagCounts, tagList, sentenceList, hyphenCounts, capitalCounts, vocab, unkValsGivenTag)
                X = run_viterbi(tagList, sentenceList, A, B)
                outputString = ""
                for i in range(len(sentenceList)):
                    outputString += sentenceList[i] + '/' + X[i] + ' '
                outputString += '\n'
                out.write(outputString)
    



    
def run_viterbi(tagList, sentenceList, A, B):

    viterbi = np.full(shape=(len(tagList)+1, len(sentenceList)), fill_value=-np.inf)
    backpointer = np.zeros(shape=(len(tagList)+1, len(sentenceList)), dtype=np.int)
    for (i, tag) in enumerate(tagList):
        viterbi[i][0] = A[0][i] + B[i][0]
        backpointer[i][0] = 0
        #print('viterbi[{}][0] = P({}|{}) + P({}|{}) = {}'.format(i, tagList[i], '<s>', sentenceList[0], tagList[i], viterbi[i][0]))
    #print('\n')
    for t in range(1, len(sentenceList)):
        #print('new t', t)

        for (s, tag0) in enumerate(tagList):
            # to get the max, we can take the matricies and add them across and get the max value
            # we add because since these are already log probs, adding makes them the eqivalent of multiplying
            maxVector = viterbi[:-1, t-1] + A[1:,s] + B[s][t]
            viterbi[s][t] = np.amax(maxVector)
            backpointer[s][t] = np.argmax(maxVector)

            #print('viterbi[{}][{}] = viterbi[{}][{}] + P({}|{}) + P({}|{}) = {} + {} + {} = {}'.format(s, t, np.argmax(maxVector), t-1, tag0, tagList[np.argmax(maxVector)], sentenceList[t], tag0, viterbi[np.argmax(maxVector)][t-1], A[np.argmax(maxVector)+1][s], B[s][t], viterbi[s][t]))
            #print('viterbi[{}][{}] = viterbi[{}][{}] + P({}|{}) + P({}|{}) = {} + {} + {} = {}'.format(s, t, np.argmax(maxVector), t-1, tag0, tagList[np.argmax(maxVector)], sentenceList[t], tag0, viterbi[np.argmax(maxVector)][t-1], A[np.argmax(maxVector)+1][s], B[s][t], viterbi[s][t]))
        #print()


    maxVector = viterbi[:, len(sentenceList)-1] + A[:,len(tagList)]
    viterbi[len(tagList), len(sentenceList) - 1] = np.amax(maxVector)
    backpointer[len(tagList), len(sentenceList) - 1] = np.argmax(maxVector)

    z = [None] * len(sentenceList)
    argmax = viterbi[0][len(sentenceList)-1]

    for k in range(1, len(tagList)):
        if viterbi[k][len(sentenceList)-1] > argmax:
            argmax = viterbi[k][len(sentenceList)-1]
            z[len(sentenceList)-1] = k
    X = [None] * len(sentenceList)
    X[len(sentenceList) - 1] = tagList[z[len(sentenceList)-1]]
    for i in range(len(sentenceList)-1, 0, -1):
        z[i-1] = backpointer[z[i]][i]
        X[i-1] = tagList[z[i-1]]
    #print(X)
    return X

    print('Finished...')

def Pkn(count, wim1wp, uniProb, sumCount):
    if (wim1wp == 0 or uniProb == 0) and max(count-discount,0)!=0 and sumCount!=0:
        prob = math.log(count-discount) - math.log(sumCount)
        return prob

    elif max(count-discount,0)==0 and uniProb != 0 and wim1wp != 0:
        prob = (math.log(discount)+math.log(wim1wp)+math.log(uniProb)) - math.log(sumCount)
        return prob
    elif uniProb != 0 and wim1wp != 0 and max(count-discount,0)!=0:
        term1 = (math.log(count-discount)) - math.log(sumCount)
        term2 = (math.log(count-discount+discount*wim1wp*uniProb)) - math.log(count-discount)
        prob = term1 + term2
        return prob
    else:
        return -np.inf


def createTransitionMatrix(transitionCounts, tagCounts, tagList):
    # we need to create a matrix of size (# of POS tags + 1) x (# of POS tags + 1)
    # this is because we only need <s> once for one part and </s> once for the other
    # add one smoothing
    A = np.zeros(shape=(len(tagList)+1, len(tagList)+1))


    # for an index A[i][j], it stores the value for P(tj | ti)

    # let's set the start states probs. This is P(ti, <s>)
    # initialize the beinning of the list
    startIndex = 0
    for (i, item) in enumerate(tagList):
        if item == '<s>':
            startIndex = i
            break


    for i in range(len(tagList)):
        count = transitionCounts.get(('<s>', tagList[i]))
        if count == None:
            count = 0

        sumCount = 0
        wim1wp = 0
        wpwi = 0
        wpwpp = 0

        if count != 0:
            wpwi = 1

        for j in range(len(tagList)):
            temp = transitionCounts.get(('<s>', tagList[j]))
            if temp == None:
                temp = 0
            else:
                wim1wp += 1
            sumCount += temp
        # special case because this is the start. wpwpp = wim1wp
        wpwpp = wim1wp



        lambdaWiMinusOne = (discount/sumCount)*wim1wp
        #lambdaWiMinusOne = (math.log(discount) - math.log(sumCount)) + math.log(wim1wp)

        PknWi = float(wpwi) / wpwpp
        #prob = max(count - discount, 0)/sumCount + lambdaWiMinusOne*PknWi
        prob=Pkn(count, wim1wp, PknWi, sumCount)

        prob = math.log(count + const) - (math.log(tagCounts[tagList[startIndex]] + const*len(tagCounts)))
        A[0,i] = prob

    # initialize the end of the list
    for i in range(1, len(tagList) + 1):
        count = transitionCounts.get((tagList[i-1], '</s>'))
        if count == None:
            count = 0
        #prob = math.log(count + const) - (math.log(tagCounts[tagList[i-1]] + const*len(tagCounts)))
        count = sumCount
        wim1wp = 0
        wpwi = 0
        wpwpp = 0

        if count != 0:
            wim1wp = 1

        for j in range(1, len(tagList) + 1):
            temp = transitionCounts.get((tagList[j-1], '</s>'))
            if temp == None:
                temp = 0
            else:
                wpwpp += 1
            sumCount += temp
        # special case because this is the start. wpwi = wpwpp
        wpwi = wpwpp

        lambdaWiMinusOne = (discount/sumCount)*wim1wp

        PknWi = float(wpwi) / wpwpp
        #prob = max(count - discount, 0)/sumCount + lambdaWiMinusOne*PknWi
        prob=Pkn(count, wim1wp, PknWi, sumCount)
        prob = math.log(count + const) - (math.log(tagCounts[tagList[i-1]] + const*len(tagCounts)))

        A[i, len(tagList)] = prob
    # initialize the middle
    for i in range(1, len(tagList) + 1):
        for j in range(len(tagList)):
            
            prev = tagList[i-1]
            curr = tagList[j]
            count = transitionCounts.get((prev,curr))

            # calculate sumCount

            if count == None:
                count = 0
            #prob = math.log(count + const) - (math.log(tagCounts[prev] + const*len(tagCounts)))

            wim1wp = 0
            wpwi = 0
            wpwpp = 0

            # handles wim1wp and sumcount
            for k in range(len(tagList)):
                temp = transitionCounts.get((prev, tagList[k]))
                if temp != None:
                    wim1wp += 1
                    sumCount += temp
            # to handle wpwi
            for k in range(len(tagList)):
                temp = transitionCounts.get((tagList[k], curr))
                if temp != None:
                    wpwi += 1
            # to handle wpwpp
            for tag0 in tagList:
                for tag1 in tagList:
                    temp = transitionCounts.get((tag0, tag1))
                    if temp != None:
                        wpwpp += 1

            lambdaWiMinusOne = (discount/sumCount)*wim1wp
            PknWi = float(wpwi) / wpwpp
            #prob = max(count - discount, 0)/sumCount + lambdaWiMinusOne*PknWi
            prob=Pkn(count, wim1wp, PknWi, sumCount)
            prob = math.log(count + const) - (math.log(tagCounts[prev] + const*len(tagCounts)))

            #print('P(ti|ti-1) = P({}|{}) = {}'.format(curr, prev, prob))
            A[i,j] = prob
    '''
    # Checking if all probs equal zero
    for i in range(len(A)):
        row_sum = np.sum(A[i])
        #print(math.pow(math.exp(1), row_sum))
        print(row_sum)
    '''

    return A

def createEmissionMatrix(observationCounts, tagCounts, tagList, sentenceList, hyphenCounts, capitalCounts, vocab, unkValsGivenTag):
    # this matrix is of size (# of POS tags) x (# of words in sentence list)
    # we are calculating P(wi|ti)
    B = np.zeros(shape=(len(tagList), len(sentenceList)))
    wptp = 0
    # calculate wptp here because we don't need to calculate it in the for loop
    '''
    for wordPrime in sentenceList:
        if wordPrime not in vocab:
            wordPrime = '<UNK>'
        for tagPrime in tagList:
            temp = observationCounts.get((wordPrime, tagPrime))
            if temp != None:
                wptp += 1
    '''

    for (j, tag) in enumerate(tagList):
        (totalCountGivenTag, unkSeenGivenTag, allSeenGivenTag) = unkValsGivenTag[tag]

        for (i, word) in enumerate(sentenceList):
            if word not in vocab:
                unkW = '<UNK>'
                if '-' in word:
                    unkW = '<UNK-HYP>'
                elif any(c.isupper() for c in word):
                    unkW = '<UNK-CAP>'
                elif any(word.endswith(s) for s in adverb_suffixes):
                    unkW = '<UNK-ADV>'
                elif any(word.endswith(s) for s in noun_suffixes):
                    unkW = '<UNK-N>'
                elif any(word.endswith(s) for s in verb_suffixes):
                    unkW = '<UNK-VRB>'
                elif any(word.endswith(s) for s in adjective_suffixes):
                    unkW = '<UNK-ADJ>'
                elif any(c.isdigit() for c in word):
                    unkW = '<UNK-NUM>'
                word = unkW


            count = observationCounts.get((word, tag))
            if count == None:
                count = 0
            '''
            witp = 0
            wpti = 0
            sumCount = 0

            # calculate witp
            for tagPrime in tagList:
                temp = observationCounts.get((word, tagPrime))
                if temp != None:
                    witp += 1
            # calculate wpti and sumcounts
            for wordPrime in sentenceList:
                temp = observationCounts.get((wordPrime, tag))
                if temp != None:
                    wpti += 1
                    sumCount += temp
            PknWi = float(witp)/wptp
            prob = Pkn(count, wpti, PknWi, sumCount)
            '''

            '''

            # witten bell smoothing?
            V = len(vocab)
            T = 0
            for wPrime in sentenceList:
                temp = observationCounts.get((wPrime, tag))
                if temp != None:
                    T += 1

            Z = V - T


            prob = 0
            if count > 0:
                prob = math.log(count) - math.log((tagCounts[tag] + T))
            else:
                if T==0 or Z==0:
                    prob = -np.inf
                else:
                    prob = math.log(T) - math.log((Z*(tagCounts[tag]+T)))
            '''

            

            hyphCount = hyphenCounts.get(tag)
            if hyphCount == None:
                hyphCount = 0
            hyphProb = math.log(hyphCount + const) - (math.log(tagCounts[tag] + const*len(hyphenCounts)))
            #hyphProb = (hyphCount+const) / (const*len(hyphenCounts))
            capCount = capitalCounts.get(tag)
            if capCount == None:
                capCount = 0
            capProb = math.log(capCount + const) - (math.log(tagCounts[tag] + const*len(capitalCounts)))
            #capProb = (capCount+const) / (const*len(capitalCounts))


            # fix the probability mass because of what we did with the unknowns
            # sum of knowns + sum of unknowns*seen*p(cap)*p(hyphen) + 
            #prob = math.log(count + const) - (math.log((tagCounts[tag]-totalCountGivenTag) + capProb*hyphProb*totalCountGivenTag + (allSeenGivenTag-unkSeenGivenTag)*const + unkSeenGivenTag*const*capProb*hyphProb))


            prob = math.log(count + const) - (math.log(tagCounts[tag] + const*(allSeenGivenTag)))



            if '<UNK' in word:
                prob += capProb + hyphProb
            

            # if a word is unknown, we can add log probs to model unknown words
            #if word == '<UNK>':
                #prob = math.log(count + const) - (math.log(tagCounts[tag] + const*len(sentenceList)))
                #if any(c.isupper() for c in word):
                    #prob = prob + capProb
                #if '-' in word:
                    #prob = prob + hyphProb
                #prob = prob + capProb + hyphProb

            #if count != 0:
            #if word == 'six':
            #print('P(w|t) = P({}|{}) = {}'.format(word, tag, prob), j,i)
            B[j][i] = prob
    '''
    for i in range(len(B)):
        row_sum = np.sum(B[i])
        print(math.pow(math.exp(1), row_sum))
    '''
    return B
        
            
            
            
    

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
