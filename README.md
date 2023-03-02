# Kneser-Ney bigram model

**Word-bigram model** with **Kneser-Ney smoothing** for next-word prediction and sentence generation.

Evaluation is done on a test set by calculating the model's perplexity.

## Examples:

1. [kn_bigram_model.py](#kn_bigram_modelpy)
2. [sentence_generator.py](#sentence_generatorpy)

### kn_bigram_model.py

**Run:**
```
> python kn_bigram_model.py data/emma_train.txt data/emma_test.txt saved/emma_model.txt
```

**Output:**
```
loading data            [##########] 100%
counting unigrams       [##########] 100%
counting bigrams        [##########] 100%
Kneser-Ney probs        [##########] 100%   
saving model            [##########] 100%

- - - - -

Sentence 1:
nobody in drawing - hour of <UNK> interest ; and nothing device presently for her .

Sentence 2:
oh !

Sentence 3:
he was good uncertainty — she recollection , miss f. but mrs. weston too it would be  
no more to be a point out of mr. elton and so for him to blindness sex should , from  
attachment any <UNK> , i was .

- - - - -
loading data            [##########] 100%

Perplexity: 79.69533547426181
```

### sentence_generator.py

**Run:**
```
> python sentence_generator.py --number 2 --trigger oh saved/emma_model.txt
```

**Output:**
```
Sentence 1:
oh !

Sentence 2:
oh no doubt of something number of vanity now .
```
