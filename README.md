# POS Tagging

Methods:
hmm - Hidden Markov Model implementation using Viterbi decoding
Infrequent words are set to be unknown words with a specific POS unknown tag based on the word's suffix

lstm - Bidirectional LSTM with CNN character embeddings
We do not use a pretrained word embedding vector, thus one is trained in here and the character embedding is appended to the word embedding. The goal of the CNN character embedding is to take into account suffix, prefix, and orthographic cues
