"""
                DISCLAIMER
        
This test class was developed by my lecturers
for educational purposes. I have included it
here for conveniency, even though it is not a
product of my own work.

                            Giulio Cusenza
"""


import unittest

from kn_bigram_model import *


class BigramTestCase(unittest.TestCase):

    def test_preprocess_sentence1(self):
        sent = "This  is \t EASY !"
        expected = ["this", "is", "easy", "!"]
        actual = preprocess_sentence(sent)
        self.assertListEqual(expected, actual)

    def test_preprocess_sentence2(self):
        sent = "This  is \t EASY !"
        expected = [BOS_MARKER, "this", "is", "easy", "!", EOS_MARKER]
        actual = preprocess_sentence(sent, markers=True)
        self.assertListEqual(expected, actual)

    def test_preprocess_sentence3(self):
        sent = "This  is \t EASY !"
        vocab = ["that", "is", "easy"]
        expected = [UNKNOWN, "is", "easy", UNKNOWN]
        actual = preprocess_sentence(sent, sorted(vocab))
        self.assertListEqual(expected, actual)

    def test_preprocess_sentence4(self):
        sent = "This  is \t EASY !"
        vocab = ["that", "is", "easy"]
        expected = [BOS_MARKER, UNKNOWN, "is", "easy", UNKNOWN, EOS_MARKER]
        actual = preprocess_sentence(sent, sorted(vocab), markers=True)
        self.assertListEqual(expected, actual)

    def test_load_data1(self):
        corpus_file = "Data/dogs.txt"
        expected = [["i", "like", "dogs"],
                    ["dogs", "like", "walks"],
                    ["she", "walks", "her", "dogs", "and", "cats"],
                    ["her", "dogs", "like", "cats"]]
        actual = load_data(corpus_file)
        self.assertListEqual(expected, actual)

    def test_load_data2(self):
        corpus_file = "Data/dogs.txt"
        vocab = ["i", "like", "dogs", "walks", "and", "cats"]
        expected = [["i", "like", "dogs"],
                    ["dogs", "like", "walks"],
                    [UNKNOWN, "walks", UNKNOWN, "dogs", "and", "cats"],
                    [UNKNOWN, "dogs", "like", "cats"]]
        actual = load_data(corpus_file, vocab=vocab)
        self.assertListEqual(expected, actual)

    def test_load_data3(self):
        corpus_file = "Data/dogs.txt"
        vocab = ["i", "like", "dogs", "walks", "and", "cats"]
        expected = [[BOS_MARKER, "i", "like", "dogs", EOS_MARKER],
                    [BOS_MARKER, "dogs", "like", "walks", EOS_MARKER],
                    [BOS_MARKER, UNKNOWN, "walks", UNKNOWN, "dogs", "and", "cats", EOS_MARKER],
                    [BOS_MARKER, UNKNOWN, "dogs", "like", "cats", EOS_MARKER]]
        actual = load_data(corpus_file, vocab=vocab, markers=True)
        self.assertListEqual(expected, actual)

    def test_get_unigram_counts1(self):
        training_data = [["i", "like", "dogs"],
                         ["dogs", "like", "walks"],
                         ["she", "walks", "her", "dogs", "and", "cats"],
                         ["her", "dogs", "like", "cats"]]
        expected_vocab = ["and", "cats", "dogs", "her", "i", "like", "she", "walks"]
        actual_vocab, _ = get_unigram_counts(training_data)
        actual_vocab = sorted(actual_vocab)
        self.assertListEqual(expected_vocab, actual_vocab)

    def test_get_unigram_counts2(self):
        training_data = [["i", "like", "dogs"],
                         ["dogs", "like", "walks"],
                         ["she", "walks", "her", "dogs", "and", "cats"],
                         ["her", "dogs", "like", "cats"]]
        expected_counts = {
            "and": 1,
            "cats": 2,
            "dogs": 4,
            "her": 2,
            "i": 1,
            "like": 3,
            "she": 1,
            "walks": 2
        }
        _, actual_counts = get_unigram_counts(training_data)
        self.assertDictEqual(expected_counts, actual_counts)

    def test_get_unigram_counts3(self):
        training_data = [["i", "like", "dogs"],
                         ["dogs", "like", "walks"],
                         ["she", "walks", "her", "dogs", "and", "cats"],
                         ["her", "dogs", "like", "cats"]]
        expected_vocab = [UNKNOWN, "cats", "dogs", "her", "like", "walks"]
        actual_vocab, _ = get_unigram_counts(training_data, remove_low_freq_words=True)
        actual_vocab = sorted(actual_vocab)
        self.assertListEqual(expected_vocab, actual_vocab)

    def test_get_unigram_counts4(self):
        training_data = [["i", "like", "dogs"],
                         ["dogs", "like", "walks"],
                         ["she", "walks", "her", "dogs", "and", "cats"],
                         ["her", "dogs", "like", "cats"]]
        expected_counts = {
            UNKNOWN: 3,
            "cats": 2,
            "dogs": 4,
            "her": 2,
            "like": 3,
            "walks": 2
        }
        _, actual_counts = get_unigram_counts(training_data, remove_low_freq_words=True)
        self.assertDictEqual(expected_counts, actual_counts)

    def test_get_vocab_index_mappings_rows(self):
        vocab = [UNKNOWN, "cats", "dogs", "her", "like", "walks"]
        expected_row_keys = sorted([UNKNOWN, BOS_MARKER, "cats", "dogs", "her", "like", "walks"])
        expected_row_values = set(range(0, 7))
        actual_row_mapping, _ = get_vocab_index_mappings(vocab)
        actual_row_keys = sorted(actual_row_mapping.keys())
        actual_row_values = set(actual_row_mapping.values())
        self.assertListEqual(expected_row_keys, actual_row_keys, msg="problem with keys in row mapping")
        self.assertSetEqual(expected_row_values, actual_row_values, msg="problem with index values in row mapping")

    def test_get_vocab_index_mappings_cols(self):
        vocab = [UNKNOWN, "cats", "dogs", "her", "like", "walks"]
        expected_col_keys = sorted([UNKNOWN, EOS_MARKER, "cats", "dogs", "her", "like", "walks"])
        expected_col_values = set(range(0, 7))
        _, actual_col_mapping = get_vocab_index_mappings(vocab)
        actual_col_keys = sorted(actual_col_mapping.keys())
        actual_col_values = set(actual_col_mapping.values())
        self.assertListEqual(expected_col_keys, actual_col_keys, msg="problem with keys in column mapping")
        self.assertSetEqual(expected_col_values, actual_col_values, msg="problem with index values in column mapping")

    def test_get_bigram_counts_no_smoothing1(self):
        training_data = [["i", "like", "dogs"],
                         ["dogs", "like", "walks"],
                         ["she", "walks", "her", "dogs", "and", "cats"],
                         ["her", "dogs", "like", "cats"]]
        row_idx = {
            BOS_MARKER: 0,
            "i": 1,
            "like": 2,
            "dogs": 3,
            "walks": 4,
            "she": 5,
            "her": 6,
            "and": 7,
            "cats": 8,
        }
        col_idx = {
            "i": 0,
            "like": 1,
            "dogs": 2,
            "walks": 3,
            "she": 4,
            "her": 5,
            "and": 6,
            "cats": 7,
            EOS_MARKER: 8
        }
        vocab = ["i", "like", "dogs", "walks", "she", "her", "and", "cats"]
        expected_matrix = np.array([
            [1, 0, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 1, 0],
            [0, 2, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2]
        ])
        bigram_matrix = get_bigram_counts(training_data, row_idx, col_idx, vocab)
        self.assertTrue(np.array_equiv(expected_matrix, bigram_matrix),
                        msg=f"\nexpected:\n{expected_matrix}\nbut got:\n{bigram_matrix}\nTest should pass even if returned matrix is of type float.")

    def test_get_bigram_counts_no_smoothing2(self):
        training_data = [["i", "like", "dogs"],
                         ["dogs", "like", "walks"],
                         ["she", "walks", "her", "dogs", "and", "cats"],
                         ["her", "dogs", "like", "cats"]]
        vocab = [UNKNOWN, "cats", "dogs", "her", "like", "walks"]
        row_idx = {
            BOS_MARKER: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5,
            UNKNOWN: 6
        }
        col_idx = {
            "cats": 0,
            "dogs": 1,
            "her": 2,
            "like": 3,
            "walks": 4,
            UNKNOWN: 5,
            EOS_MARKER: 6
        }
        expected_matrix = np.array([
            [0, 1, 1, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 2, 0, 1, 1],
            [0, 2, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0]
        ])
        bigram_matrix = get_bigram_counts(training_data, row_idx, col_idx, vocab)
        self.assertTrue(np.array_equiv(expected_matrix, bigram_matrix),
                        msg=f"\nexpected:\n{expected_matrix}\nbut got:\n{bigram_matrix}\nTest should pass even if returned matrix is of type float.")

    def test_get_bigram_counts_laplace_smoothing1(self):
        training_data = [["i", "like", "dogs"],
                         ["dogs", "like", "walks"],
                         ["she", "walks", "her", "dogs", "and", "cats"],
                         ["her", "dogs", "like", "cats"]]
        row_idx = {
            BOS_MARKER: 0,
            "i": 1,
            "like": 2,
            "dogs": 3,
            "walks": 4,
            "she": 5,
            "her": 6,
            "and": 7,
            "cats": 8,
        }
        col_idx = {
            "i": 0,
            "like": 1,
            "dogs": 2,
            "walks": 3,
            "she": 4,
            "her": 5,
            "and": 6,
            "cats": 7,
            EOS_MARKER: 8
        }
        vocab = ["i", "like", "dogs", "walks", "she", "her", "and", "cats"]
        expected_matrix = np.array([
            [2, 1, 2, 1, 2, 2, 1, 1, 1],
            [1, 2, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 2, 2, 1, 1, 1, 2, 1],
            [1, 3, 1, 1, 1, 1, 2, 1, 2],
            [1, 1, 1, 1, 1, 2, 1, 1, 2],
            [1, 1, 1, 2, 1, 1, 1, 1, 1],
            [1, 1, 3, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 3]
        ])
        bigram_matrix = get_bigram_counts(training_data, row_idx, col_idx, vocab, laplace=True)
        self.assertTrue(np.array_equiv(expected_matrix, bigram_matrix),
                        msg=f"\nexpected:\n{expected_matrix}\nbut got:\n{bigram_matrix}\nTest should pass even if returned matrix is of type float.")

    def test_get_bigram_counts_laplace_smoothing2(self):
        training_data = [["i", "like", "dogs"],
                         ["dogs", "like", "walks"],
                         ["she", "walks", "her", "dogs", "and", "cats"],
                         ["her", "dogs", "like", "cats"]]
        vocab = [UNKNOWN, "cats", "dogs", "her", "like", "walks"]
        row_idx = {
            BOS_MARKER: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5,
            UNKNOWN: 6
        }
        col_idx = {
            "cats": 0,
            "dogs": 1,
            "her": 2,
            "like": 3,
            "walks": 4,
            UNKNOWN: 5,
            EOS_MARKER: 6
        }
        expected_matrix = np.array([
            [1, 2, 2, 1, 1, 3, 1],
            [1, 1, 1, 1, 1, 1, 3],
            [1, 1, 1, 3, 1, 2, 2],
            [1, 3, 1, 1, 1, 1, 1],
            [2, 2, 1, 1, 2, 1, 1],
            [1, 1, 2, 1, 1, 1, 2],
            [2, 1, 1, 2, 2, 1, 1]
        ])
        bigram_matrix = get_bigram_counts(training_data, row_idx, col_idx, vocab, laplace=True)
        self.assertTrue(np.array_equiv(expected_matrix, bigram_matrix),
                        msg=f"\nexpected:\n{expected_matrix}\nbut got:\n{bigram_matrix}\nTest should pass even if returned matrix is of type float.")

    def test_counts_to_probs(self):
        bigram_counts = np.array([
            [1, 2, 2],
            [1, 3, 1],
            [2, 1, 1]
        ])
        expected = np.array([
            [1/5, 2/5, 2/5],
            [1/5, 3/5, 1/5],
            [2/4, 1/4, 1/4]
        ])
        actual = counts_to_probs(bigram_counts)
        self.assertTrue(np.array_equiv(expected, actual),
                        msg=f"\nexpected:\n{expected}\nbut got:\n{actual}")

    def test_to_log10(self):
        bigram_probs = np.array([
            [1/5, 2/5, 2/5],
            [1/5, 3/5, 1/5],
            [2/4, 1/4, 1/4]
        ])
        expected = np.array([
             [-0.6989, -0.3979, -0.3979],
             [-0.6989, -0.2218, -0.6989],
             [-0.3010, -0.6020, -0.6020]
        ])
        actual = to_log10(bigram_probs)
        self.assertTrue(np.allclose(expected, actual, atol=.0001),
                        msg=f"\nexpected:\n{expected}\nbut got:\n{actual}")

    def test_generate_sentence(self):
        training_data = [["i", "like", "dogs"],
                         ["dogs", "like", "walks"],
                         ["she", "walks", "her", "dogs", "and", "cats"],
                         ["her", "dogs", "like", "cats"]]
        vocab = [UNKNOWN, "cats", "dogs", "her", "like", "walks"]
        row_idx = {
            BOS_MARKER: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5,
            UNKNOWN: 6
        }
        col_idx = {
            "cats": 0,
            "dogs": 1,
            "her": 2,
            "like": 3,
            "walks": 4,
            UNKNOWN: 5,
            EOS_MARKER: 6
        }

        bigram_probs = np.array([
        [0.09090909, 0.18181818, 0.18181818, 0.09090909, 0.09090909, 0.27272727, 0.09090909],
        [0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.33333333],
        [0.09090909, 0.09090909, 0.09090909, 0.27272727, 0.09090909, 0.18181818, 0.18181818],
        [0.11111111, 0.33333333, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111],
        [0.2, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1],
        [0.11111111, 0.11111111, 0.22222222, 0.11111111, 0.11111111, 0.11111111, 0.22222222],
        [0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1]
        ])

        for i in range(20):
            sentence = generate_sentence(bigram_probs, row_idx, col_idx)
            self.assertEqual(BOS_MARKER, sentence[0],
                             msg="Sentences should start with BOS.")
            self.assertEqual(EOS_MARKER, sentence[len(sentence)-1],
                             msg="Sentences should end with EOS.")

    def test_get_sent_logprob(self):
        row_idx = {
            BOS_MARKER: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5,
            UNKNOWN: 6
        }
        col_idx = {
            "cats": 0,
            "dogs": 1,
            "her": 2,
            "like": 3,
            "walks": 4,
            UNKNOWN: 5,
            EOS_MARKER: 6
        }
        bigram_logprobs = np.array([
            [-1.04139269, -0.74036269, -0.74036269, -1.04139269, -1.04139269, -0.56427143, -1.04139269],
            [-0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.47712126],
            [-1.04139269, -1.04139269, -1.04139269, -0.56427143, -1.04139269, -0.74036269, -0.74036269],
            [-0.95424251, -0.47712126, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251],
            [-0.69897, -0.69897, -1., -1., -0.69897, -1., -1.],
            [-0.95424251, -0.95424251, -0.65321252, -0.95424251, -0.95424251, -0.95424251, -0.65321252],
            [-0.69897, -1., -1., -0.69897, -0.69897, -1., -1.]
        ])
        sent = [BOS_MARKER, "cats", "like", "dogs", EOS_MARKER]
        expected = -3.43496789
        actual = get_sent_logprob(bigram_logprobs, row_idx, col_idx, sent)
        self.assertAlmostEqual(expected, actual)

    def test_get_perplexity1(self):
        row_idx = {
            BOS_MARKER: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5,
            UNKNOWN: 6
        }
        col_idx = {
            "cats": 0,
            "dogs": 1,
            "her": 2,
            "like": 3,
            "walks": 4,
            UNKNOWN: 5,
            EOS_MARKER: 6
        }
        bigram_logprobs = np.array([
            [-1.04139269, -0.74036269, -0.74036269, -1.04139269, -1.04139269, -0.56427143, -1.04139269],
            [-0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.47712126],
            [-1.04139269, -1.04139269, -1.04139269, -0.56427143, -1.04139269, -0.74036269, -0.74036269],
            [-0.95424251, -0.47712126, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251],
            [-0.69897, -0.69897, -1., -1., -0.69897, -1., -1.],
            [-0.95424251, -0.95424251, -0.65321252, -0.95424251, -0.95424251, -0.95424251, -0.65321252],
            [-0.69897, -1., -1., -0.69897, -0.69897, -1., -1.]
        ])
        test_sentences = [
            [BOS_MARKER, "cats", "like", "dogs", EOS_MARKER],
            [BOS_MARKER, "dogs", UNKNOWN, "cats", "like", "her", EOS_MARKER]
        ]
        expected = 7.117292739769333
        actual = get_perplexity(bigram_logprobs, row_idx, col_idx, test_sentences)
        self.assertAlmostEqual(expected, actual)

    def test_get_perplexity2(self):
        row_idx = {
            BOS_MARKER: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5,
            UNKNOWN: 6
        }
        col_idx = {
            "cats": 0,
            "dogs": 1,
            "her": 2,
            "like": 3,
            "walks": 4,
            UNKNOWN: 5,
            EOS_MARKER: 6
        }
        bigram_logprobs = np.array([
            [-1.04139269, -0.74036269, -0.74036269, -1.04139269, -1.04139269, -0.56427143, -1.04139269],
            [-0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.47712126],
            [-1.04139269, -1.04139269, -1.04139269, -0.56427143, -1.04139269, -0.74036269, -0.74036269],
            [-0.95424251, -0.47712126, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251],
            [-0.69897, -0.69897, -1., -1., -0.69897, -1., -1.],
            [-0.95424251, -0.95424251, -0.65321252, -0.95424251, -0.95424251, -0.95424251, -0.65321252],
            [-0.69897, -1., -1., -0.69897, -0.69897, -1., -1.]
        ])
        test_sentence = [
            [BOS_MARKER, "cats", "like", "dogs", EOS_MARKER],
        ]
        expected = 7.223405110664793
        actual = get_perplexity(bigram_logprobs, row_idx, col_idx, test_sentence)
        self.assertAlmostEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
