"""
                HONOR CODE
        
All the code inside the methods of this class is
a product of my own work on a pattern provided
by my lecturers. The pattern included only the
method declarations and docstring descriptions.

For this Kneser-Ney version I have edited in some
cases the provided pattern, as the basic one was
intended for add-1 smoothing. To speed up K.-N.
smoothing, I have also done a few performance
improvements, and added methods such as binary
search.

The method save_model() is fully mine.

                                Giulio Cusenza
"""

import argparse
import numpy as np

# use binary search to speed up vocabulary operations
import bisect
from bisect import bisect_left

BOS_MARKER = "<s>"
EOS_MARKER = "</s>"
UNKNOWN = "<UNK>"

# random number generator
rng = np.random.default_rng()


def binary_search(seq, item):
    """
    Search for an item in a sequence through binary search.

    :seq: sequence to search in
    :item: item to be searched for
    :return: the index of the item, or -1 if it was not found.
    """
    i = bisect_left(seq, item)
    if i != len(seq) and seq[i] == item:
        return i
    else:
        return -1


def preprocess_sentence(sent, vocab=None, markers=False):
    """
    Preprocess sent by tokenizing (splitting on whitespace),
    and converting tokens to lowercase. If vocab is provided, OOV words
    are replaced with UNKNOWN. If markers is True, sentence begin and
    end markers are inserted.

    :param sent: sentence to process, as a string
    :param vocab: if provided, oov words are replaced with UNKNOWN
    :param markers: if True, BOS and EOS markers are added
    :return: list of lowercase token strings, with optional OOV replacement
    and marker insertion
    """
    sentence = []
    
    # if a vocabulary is provided...
    if vocab:
        for token in str(sent).split(): # split by the whitespaces
            token = token.lower()
            if binary_search(vocab, token) != -1:  # if the token is in the vocabulary, save it     
                sentence.append(token)
            else:
                sentence.append(UNKNOWN)  # otherwise add an UNKNOWN token
    
    # if no vocabulary is provided, just add the tokens
    else:
        for token in str(sent).split():
            sentence.append(token.lower())
    
    # return (with markers if markers is true)
    if markers:
        return [BOS_MARKER] + sentence + [EOS_MARKER]
    else:
        return sentence


def load_data(corpus_file, vocab=None, markers=False):
    """
    Read corpus file line-by-line and return a list of preprocessed sentences,
    where preprocessing includes tokenization and lowercasing tokens. If vocab
    is provided, OOV words are replaced with the UNKNOWN token. If markers is True,
    sentence begin and end markers are inserted.

    Notes:

    - vocab can only be provided when loading test data, after the vocabulary has
    been established

    - markers should be False when loading training data (the markers will be added later,
    in get_bigram_counts(), after the unigram counts have been calculated),
    and True when loading test data.

    :param corpus_file: file containing one sentence per line
    :param vocab: if provided, OOV words are replaced with UNKNOWN
    :param markers: if True, BOS and EOS markers are added
    :return: a list of lists representing preprocessed sentences
    """
    # sort vocabulary to use binary search in preprocess_sentence()
    if vocab:
       vocab = sorted(vocab)

    # count total number of sentences for visual feedback
    n_sentences = 0
    with open(corpus_file, encoding="utf-8") as corpus:
        for sentence in corpus:
            n_sentences += 1
    
    # process sentences
    sentences = []
    with open(corpus_file, encoding="utf-8") as corpus:
        progress = 0
        for sentence in corpus:
            # call preprocess_sentence() on each sentence in the corpus
            sentences.append(preprocess_sentence(sentence, vocab, markers))

            # visual feedback
            if progress % (n_sentences // 100) == 0:
                percent = int(progress / n_sentences * 100)
                done = percent // 10
                todo = 10 - done
                print("\rloading data\t\t[" + "#"*done + "-"*todo + "] " + str(percent) + "%", end="")
            progress += 1
        print("\rloading data\t\t[##########] 100%")
        
    return sentences


def get_unigram_counts(training_data, remove_low_freq_words=False, freq_threshold=2):
    """
    From the training data, get the vocabulary as a list of words,
    and the unigram counts represented as a dictionary with
    words as keys and frequencies as values. If remove_low_freq_words
    is True, any word with count == 1 is removed
    from the dictionary, and its count (1) is added to the count of
    the UNKNOWN token.

    :param training_data: list of lists of words, without sentence markers
    :param remove_low_freq_words: if True, transfer the count of words appearing
    only once to the UNKNOWN  token
    :param freq_threshold: how often does a word need to occur to not be replaced with <UNK>
    :return: a list of vocabulary words, and a dictionary of words:frequencies (optionally
    with low-frequency word counts transferred to the UNKNOWN token)
    """
    # initialise output variables
    types = []
    type_freq = {UNKNOWN: 0}
    
    # go through all the tokens in the data
    progress = 0
    for sentence in training_data:
        for token in sentence:
            # save types
            if binary_search(types, token) == -1:
                bisect.insort(types, token)
                type_freq.update({token: 1})
            else:
                type_freq[token] += 1
                
        # visual feedback
        if progress % (len(training_data) // 100) == 0:
            percent = int(progress / len(training_data) * 95)
            done = percent // 10
            todo = 10 - done
            print("\rcounting unigrams\t[" + "#"*done + "-"*todo + "] " + str(percent) + "%", end="")
        progress += 1
    
    # only after the tokens have been counted, those
    # with freq =< 1 can be replaced by UNKNOWN     
    if remove_low_freq_words:
        removed = False
        progress = 0
        new_types = []
        new_type_freq = {UNKNOWN: 0}
        unknown = new_type_freq[UNKNOWN]
        for type in types:
            freq = type_freq[type]
            if freq >= freq_threshold:
                # keep type
                new_types.append(type)
                new_type_freq.update({type : freq})
            else:
                # skip type and increment the freq of the UNKNOWN type
                unknown += 1
                removed = True
                
            # visual feedback
            if progress % (len(training_data) // 100) == 0:
                percent = 95 + int(progress / len(training_data) * 5)
                done = percent // 10
                todo = 10 - done
                print("\rcounting unigrams\t[" + "#"*done + "-"*todo + "] " + str(percent) + "%", end="")
            progress += 1
                 
        # if something has been removed, we need to add the
        # UNKNOWN type to types if it is not already there.
        types = new_types
        if removed and binary_search(types, UNKNOWN) == -1:
            types.append(UNKNOWN)
    
    print("\rcounting unigrams\t[##########] 100%")  
    return types, type_freq


def get_vocab_index_mappings(vocab):
    """
    Assign each word in the vocabulary an index for access into the bigram probability
    matrix. Create and return 2 mappings, one for the rows and one for the
    columns. The mappings are dictionaries with vocabulary words as keys, and
    indices as values. The indices should start at 0 and each word should have
    a unique index. Include a BOS index in the row mapping, and an EOS index
    in the column mapping.

    :param vocab: a list of vocabulary words
    :return: two dictionaries with index mappings
    """
    # initialise dictionaries
    word_indices_rows = {BOS_MARKER: 0}
    word_indices_cols = {}
    
    # fill in dictionaries
    for index, word in enumerate(vocab):
        # on the rows there is already the BOS_MARKER, so everything is shifted by one
        word_indices_rows.update({word: index+1})
        # on the columns no particular treatment
        word_indices_cols.update({word: index})
        
    # add EOS_MARKER to the columns
    word_indices_cols.update({EOS_MARKER: len(vocab)})

    return word_indices_rows, word_indices_cols
    
    
def get_bigram_counts(training_data, row_idx, col_idx, vocab, laplace=False):
    """
    Create and return a 2D matrix containing the bigram counts in training_data,
    optionally using Laplace smoothing.

    Before updating the bigrams counts for a sentence:

    - replace words in the sentence that are not in the vocabulary with the UNKNOWN token
    - add BOS and EOS sentence markers

    Note: It is possible to have words in the training data that are not part of the
    vocabulary if remove_low_freq_words=True when calling get_unigram_counts().

    :param training_data: list of lists of words, without sentence markers
    :param row_idx: word:row_index mapping
    :param col_idx: word:col_index mapping
    :param vocab: list of vocabulary words
    :param laplace: if True, use Laplace smoothing
    :return: a 2D matrix containing the bigram counts, optionally with Laplace smoothing
    """
    # sort vocabulary to use binary search in preprocess_sentence()
    if vocab:
       vocab = sorted(vocab)
       
    # initialise matrix
    bigram_counts = np.zeros((len(row_idx), len(col_idx)), dtype=np.float16)
    
    # count bigrams
    progress = 0
    for sentence in training_data:
        sentence = preprocess_sentence(' '.join(sentence), vocab, markers=True)
        for i in range(len(sentence)-1):
            row_i = row_idx[sentence[i]]
            col_i = col_idx[sentence[i+1]]
            bigram_counts[row_i][col_i] += 1
            
        # visual feedback
        if progress % (len(training_data) // 100) == 0:
            percent = int(progress / len(training_data) * 100)
            done = percent // 10
            todo = 10 - done
            print("\rcounting bigrams\t[" + "#"*done + "-"*todo + "] " + str(percent) + "%", end="")
        progress += 1
    
    # la Place smoothing
    if laplace:
        bigram_counts += 1
    
    print("\rcounting bigrams\t[##########] 100%")      
    return bigram_counts


def counts_to_probs(bigram_counts):
    """
    Returns a 2D matrix containing the bigram probabilities.

    :param bigram_counts: 2D matrix of integer counts of bigrams
    :return: a 2D matrix containing the bigram probabilities
    """
    # initialise matrix
    bigram_probs = np.zeros((len(bigram_counts), len(bigram_counts[0])), dtype=np.float16)
    
    # fill in matrix
    for i in range(len(bigram_probs)):
        bigram_probs[i] = bigram_counts[i] / np.sum(bigram_counts[i])
            
    return bigram_probs


def kn_probs(bigram_counts, d=0.75):
    """
    Calculate Kneser-Ney probabilities on a matrix of bigram counts.
    
    :param bigram_counts: 2D matrix of integer counts of bigrams
    :param d: discount value subtracted from the count of each bigram
    :return: a 2D matrix containing the Kneser-Ney bigram probabilities
    """
    print("training\t\t[----------] 0%", end="", flush=True)
    # initialise matrix
    length = len(bigram_counts)
    kn_probs = np.zeros((length, len(bigram_counts[0])), dtype=np.float16)
    
    # count bigram types
    # count the row and colum types
    n_bigram_types = 0
    row_types_counts = [0] * length
    col_types_counts = [0] * length
    for row_i, row in enumerate(bigram_counts):
        for col_i, count in enumerate(row):
            if count > 0:
                n_bigram_types += 1
                row_types_counts[row_i] += 1
                col_types_counts[col_i] += 1
                
        # training visual feedback
        if row_i % (length // 100) == 0:
            percent = round(row_i / length * 50, 3)
            done = int(percent // 10)
            todo = 10 - done
            print("\rtraining\t\t[" + "#"*done + "-"*todo + "] " + "{:.3f}".format(percent) + "%", end="", flush=True)
    
    # iterate through the matrix
    for row_i, row in enumerate(bigram_counts):
        
        # compute the pre_word (i.e. row) terms
        pre_word_count = np.sum(row)
        n_pre_word_followers = row_types_counts[row_i]
        kn_lambda = (d / pre_word_count) * n_pre_word_followers
        
        # iterate through the row
        for col_i, bigram_count in enumerate(row):
            
            n_word_precedents = col_types_counts[col_i]
            
            # compute KN prob and save it
            kn_prob = (max(bigram_count-d, 0) / pre_word_count)\
                        + (kn_lambda * (n_word_precedents / n_bigram_types))
            kn_probs[row_i][col_i] = kn_prob
            
        # training visual feedback
        if row_i % (length // 100) == 0:
            percent = 50 + round(row_i / length * 50, 3)
            done = int(percent // 10)
            todo = 10 - done
            print("\rtraining\t\t[" + "#"*done + "-"*todo + "] " + "{:.3f}".format(percent) + "%", end="", flush=True)
    print("\rtraining\t\t[##########] 100%   ")
    
    return kn_probs


def to_log10(bigram_probs):
    """
    Convert a probability matrix to a log 10 probability matrix.

    :param bigram_probs: probability matrix
    :return: log 10 probability matrix
    """
    return np.log10(bigram_probs)


def generate_sentence(bigram_probs, row_idx, col_idx, start=BOS_MARKER):
    """
    Generate a sentence with probabilities according to the distributions
    given by bigram_probs. The returned sentence is a list of words that
    starts with BOS and continues until EOS is generated.

    :param bigram_probs: bigram probability matrix (not log10 matrix)
    :param row_idx: index mapping for rows
    :param col_idx: index mapping for columns
    :param start: optional word to start the sentence generation
    :return: a sentence as a list of words, generated using bigram_probs
    """
    # initialise sentence
    if start != BOS_MARKER:
        sentence = [BOS_MARKER, start]
    else:
        sentence = [BOS_MARKER]
    
    # continue until we reach the end marker...
    while (sentence[-1] != EOS_MARKER):
        # based on the last word in the sentence, take the row with
        # the candidates' probabilities for following that word
        row_probs = bigram_probs[row_idx[sentence[-1]]]
        # get the index of a random candidate out of the best ones
        best_candidate_idx = rng.choice(range(len(row_probs)), p=row_probs)
        # retrieve best candidate word from the index and append it to the end of the sentence
        best_candidate = list(col_idx.keys())[best_candidate_idx]
        sentence.append(best_candidate)
        
    return sentence
            

def get_sent_logprob(bigram_logprobs, row_idx, col_idx, sent):
    """
    Returns the log 10 probability of sent.

    :param bigram_logprobs: log10 bigram matrix
    :param row_idx: row index mapping
    :param col_idx: column index mapping
    :param sent: a preprocessed sentence with BOS and EOS markers
    :return: the log10 probability of sent
    """
    logprob = 0
    # retrieve the bigram probabilities from the matrix and sum them together
    for i in range(len(sent)-1):
        row_i = row_idx[sent[i]]
        col_i = col_idx[sent[i+1]]
        # NB: summing up log probabilities is analogue to multiplying normal probabilities
        logprob += bigram_logprobs[row_i][col_i]
        
    return logprob


def get_perplexity(bigram_logprobs, row_idx, col_idx, test_sentences):
    """
    Calculate the perplexity of the test sentences according to the given
    bigram model.

    Notes:

    - Keep in mind that log10 probabilities are given, to avoid underflow,
    which means that the formula needs to be adjusted.

    - Get the perplexity of each sentence individually

    - Be careful with calculating N - see J&M 3.2.1

    :param bigram_logprobs: bigram logprob matrix
    :param row_idx: row index mapping
    :param col_idx: column index mapping
    :param test_sentences: list of preprocessed test sentences with BOS and EOS markers
    :return: the perplexity of the test sentences according to the given bigram model
    """
    total_logprob = 0
    n_bigrams = 0
    for sentence in test_sentences:
        total_logprob += get_sent_logprob(bigram_logprobs, row_idx, col_idx, sentence)
        n_bigrams += len(sentence)-1
    
    return 10**(-(1/n_bigrams)*(total_logprob))


def save_model(matrix, vocab, file_name): # TODO try to use numpy save to file methods instead
    """
    Save model to a file (vocabulary words one for each line, then a blank
    line, then the csv matrix).
    
    :param matrix: 2D matrix to be saved
    :param vocab: list of types (needed to remap the matrix indices to words)
    :param file_name: name of the file that will store the matrix
    """
    print("\rsaving model\t\t[----------] 0%", end="")
    
    # save vocabulary
    with open(file_name, "w", encoding="utf-8") as file:
        for word in vocab:
            file.write(word + "\n")
        file.write("\n\n")
    
    # save matrix
    with open(file_name, "a", encoding="utf-8") as file:
        progress = 0
        for row in matrix:
            for col in row:
                file.write(str(col) + ",")
            file.write("\n")
            
            # saving visual feedback
            if progress % (len(matrix) // 100) == 0:
                percent = int(progress / len(matrix) * 100)
                done = percent // 10
                todo = 10 - done
                print("\rsaving model\t\t[" + "#"*done + "-"*todo + "] " + str(percent) + "%", end="")
            progress += 1
    print("\rsaving model\t\t[##########] 100%")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_file", help="corpus file")
    parser.add_argument("test_file", help="test file")
    parser.add_argument("save_file", help="save file")
    return parser.parse_args()


def main(args):
    """
    Train a bigram model on a file given by args.corpus_file. Remove low-frequency
    words, and apply Laplace smoothing.

    Generate and print 3 sentences.

    Evaluate your model by calculating the perplexity of the test data
    in args.test_file. Print the perplexity value.

    :param args: command-line arguments (args.corpus_file, args.test_file)
    """
    # load data with no sentence markings
    training_data = load_data(args.corpus_file)
    
    # construct vocabulary and mappings
    vocab, unigram_counts = get_unigram_counts(training_data, remove_low_freq_words=True, freq_threshold=3)
    del(unigram_counts) # not needed for this bigram model
    row_idx, col_idx = get_vocab_index_mappings(vocab)
    
    # compute probability matrices
    bigram_counts = get_bigram_counts(training_data, row_idx, col_idx, vocab)
    del(training_data)
    bigram_probs = kn_probs(bigram_counts)
    
    # save model
    save_model(bigram_probs, vocab, args.save_file)
    del(bigram_counts)
    
    # generate three sentences
    print("\n- - - - -")
    for i in range(1, 4):
        print("\nSentence " + str(i) + ":")
        print(" ".join(generate_sentence(bigram_probs, row_idx, col_idx)[1:-1]))
    print("\n- - - - -")
        
    # evaluate model
    test_data = load_data(args.test_file, vocab, markers=True)
    del(vocab)
    bigram_logprobs = to_log10(bigram_probs)
    del(bigram_probs)
    print("\nPerplexity:", get_perplexity(bigram_logprobs, row_idx, col_idx, test_data))


if __name__ == '__main__':
    main(parse_args())