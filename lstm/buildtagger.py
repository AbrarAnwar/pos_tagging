# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MAX_LEN = 64 # the highest in our data was 54, but powers of 2 are efficient

class POSNet(nn.Module):
    
    def __init__(self, w_emb_dim, char_emb_dim, w_emb_output_dim, char_emb_output_dim, vocab_size, tags_size, chars_size):
        super(POSNet, self).__init__()

        # create all the layers we need to be defined as class variables
        # this creates the general word embeddings
        self.word_embedding = nn.Embedding(vocab_size, w_emb_dim)

        # now we create the character embeddings
        self.char_embedding = nn.Embedding(chars_size, char_emb_dim)
        self.char_cnn = nn.Conv1d(char_emb_dim, char_emb_output_dim, 5)
        self.max_pool = nn.MaxPool1d(3)

        # now we can create the lstm
        self.lstm = nn.LSTM(input_size=w_emb_dim + char_emb_output_dim, hidden_size=w_emb_output_dim, bidirectional=True)

        # put through linear layer
        self.linLayer = nn.Linear(2*w_emb_output_dim, tags_size)

    def forward(self, sentence, words):

        embeds = self.word_embedding(sentence)

        cnn_word_reps = []
        # get the cnn reps per word in the sentence
        word_char_embs = []

        for word in words:
            char_embeddings = self.char_embedding(word).transpose(1,0)
            word_char_embs.append(char_embeddings)
        # make this of size sentence length x embed_size x MAX_LEN
        word_char_embs = torch.stack(tuple(word_char_embs))

        # output size should be sentence length x channels (128) x output size (128)
        cnn_output = self.char_cnn(word_char_embs)

        # now we need to maxpool it. (doing it across dimensions to get proper size
        #cnn_output = self.max_pool(cnn_output)
        cnn_output, _ = torch.max(cnn_output, 2)


            #print(char_embeddings.view(1, 128, 64).shape)
            # be sure to reshape input to proper size with view
            #cnn_output = self.char_cnn(char_embeddings.view(1,128,64))
            #cnn_output = self.max_pool(cnn_output)
            #print(cnn_output.shape)
            #cnn_output = cnn_output.view(1024, -1)


            #cnn_word_reps.append(cnn_output)

        #cnn_word_reps = torch.stack(tuple(cnn_word_reps))

        #embeds = embeds.view(-1, 1024, 1)

        # we have to concatenate word embeddings and the cnn embeddings
        concat = torch.cat((embeds, cnn_output), dim=-1).unsqueeze(0)


        # put this through our lstm
        hidden_reps, tup = self.lstm(concat)
        output =  self.linLayer(hidden_reps.view(len(sentence), -1))

        probs = F.log_softmax(output, dim=1)
        return probs


# returns a unique integer index for each word w
def idx(w, vocab):
    if w in vocab.keys():
        return vocab[w]
    # not sure if this will work but let's try it
    # this is how we handle unknowns

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    
    # build vocabulary here
    tags = {}
    vocab = {}
    chars = {}
    with open(train_file) as train_sentences:
        for line in train_sentences:
            for wordAndTag in line.split():
                indx = wordAndTag.rfind('/')
                currTag = wordAndTag[indx+1:]
                currWord = wordAndTag[:indx]
                # doing this provides a unique integer index for each word
                if currWord not in vocab.keys():
                    vocab[currWord] = len(vocab)
                
                for char in currWord:
                    if char not in chars.keys():
                        chars[char] = len(chars)


    with open(train_file) as train_sentences:
        for line in train_sentences:
            #prevWord = '<s>'
            #prevTag = '<s>'
            #tags[prevTag] = tagCounts.get(prevTag, 0) + 1
            for wordAndTag in line.split():
                indx = wordAndTag.rfind('/')
                currTag = wordAndTag[indx+1:]
                currWord = wordAndTag[:indx]
                if currTag not in tags:
                    tags[currTag] = len(tags)

                prevWord = currWord
                prevTag = currTag

            #tags['</s>'] = tagCounts.get('</s>', 0) + 1
    maxLen = 0
    maxW = None
    for w in vocab:
        if len(w) > maxLen:
            maxLen = len(w)
            maxW = w

    



    # time to start training our model
    # w_emb_dim, char_emb_dim, w_emb_output_dim, char_emb_output_dim
    model = POSNet(1024, 128, 1024, 1024, len(vocab), len(tags), len(chars)+1)
    # set these so it works on test machines
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # set as stated by pa2 docs
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=.01)

    print("starting training...")
    for epoch in range(100):
        acc = 0
        with open(train_file) as train_sentences:
            for i, line in enumerate(train_sentences):
                wordList = []
                tagList = []
                # get these ready for training
                for wordAndTag in line.split():
                    indx = wordAndTag.rfind('/')
                    currTag = wordAndTag[indx+1:]
                    currWord = wordAndTag[:indx]
                    wordList.append(currWord)
                    tagList.append(currTag)
                # we will now tensorify the words
                word_tensors = []
                for w in wordList:
                    word_tensors.append(idx(w, vocab))
                # we also now need the character-level representations
                word_tensors = torch.LongTensor(tuple(word_tensors)).to(device)




                word_char_tensors = []
                for w in wordList:
                    char_rep = []
                    for char in w:
                        char_rep.append(torch.tensor(idx(char, chars), dtype=torch.long).to(device))
                    for j in range(len(char_rep), MAX_LEN):
                        char_rep.append(torch.tensor(84, dtype=torch.long).to(device))
                    word_char_tensors.append(torch.tensor(char_rep, dtype=torch.long).to(device))
                # lastly we need the targets
                tag_tensors = []
                for t in tagList:
                    x = torch.zeros(len(tags), dtype=torch.long).to(device)
                    x[idx(t,tags)] = 1
                    tag_tensors.append(x)
                tag_tensors = torch.stack((tag_tensors)).to(device)

                model.zero_grad()



                probs = model(word_tensors, word_char_tensors)
                
                # have to do torch.max because CrossEntropyLoss expects an index,
                # not a one hot encoding
                loss = criterion(probs, torch.max(tag_tensors, 1)[1])
                loss.backward()
                optimizer.step()
                loss += loss.item()
                _, indices = torch.max(probs, 1)
                acc += torch.mean(torch.tensor(torch.max(tag_tensors,1)[1] == indices, dtype=torch.float))
                if i % 500 == 0:
                    print(torch.mean(torch.tensor(torch.max(tag_tensors,1)[1] == indices, dtype=torch.float)))
                    print("Epoch {} Running;\t{}% Complete, loss {}".format(epoch + 1, i/39832.0, loss))

        loss = loss/39832.0
        acc = acc/39832.0
        print("Epoch {} Completed;\tLoss={}\tAccuracy={}".format(epoch, loss, acc))
                





                    


    print('Finished...')
		
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
