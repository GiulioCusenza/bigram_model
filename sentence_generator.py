"""
                HONOR CODE
        
All the code inside the methods of this class is
a product of my own work.
                                Giulio Cusenza
"""

import argparse
import numpy as np
import kn_bigram_model as kn

BOS_MARKER = "<s>"
EOS_MARKER = "</s>"
UNKNOWN = "<UNK>"

# random number generator
rng = np.random.default_rng()


def load_model(file):
    """
    Load a bigram model from a csv file.

    :param file: model file (vocabulary words one for each line,
    then a blank line, then the csv matrix).
    :return: model matrix, row mappings, column mappings.
    """
    with open(file, "r", encoding="utf-8") as file:
        
        # load vocabulary words
        vocab = []
        for line in file:
            # if the line is empty, stop reading words and pass to matrix
            if not line.strip("\n"):
                break
            vocab.append(line.strip())
        
        # initialise matrix
        row_idx, col_idx = kn.get_vocab_index_mappings(vocab)
        model = []
        
        # continue line iteration and load matrix
        for line in file:
            if not line.strip("\n"):
                continue
            row = np.array(line.strip()[:-1].split(","), dtype=float)
            normalized_row = row * (1/sum(row))
            model.append(normalized_row)

    return model, row_idx, col_idx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model")
    parser.add_argument("--number", "-n", type=int, help="number of sentences to generate")
    parser.add_argument("--trigger", "-t", help="a trigger word to start your sentence")
    return parser.parse_args()


def main(args):
    """
    Generate sentence.

    :args: command-line arguments (args.model and args.trigger)
    """
    # load model
    model, row_idx, col_idx = load_model(args.model)
    
    # set trigger
    if args.trigger:
        if args.trigger in row_idx:
            start = args.trigger
        else:
            print("The given trigger \"" + args.trigger + "\" is not known to the model.")
            exit()
    else:
        start = BOS_MARKER
    
    # generate sentence
    print()
    for i in range(args.number):
        print("Sentence " + str(i+1) + ":")
        print(" ".join(kn.generate_sentence(model, row_idx, col_idx, start)[1:-1]), "\n")
    
    
if __name__ == '__main__':
    main(parse_args())