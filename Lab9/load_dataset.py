import glob
import numpy as np
import os.path
import re
import pickle

pickle_file = 'data/cache_data.pkl'
numpy_file = 'data/numpy_data.npy'

def load_from_pickle():
    print('Opening from pickle cache')
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    with open(numpy_file, "rb") as f:
        network_input = np.load(f)
        network_output = np.load(f)
    return network_input, network_output, data[0], data[1], data[2]

def save_to_pickle(network_input, network_output, words, indices_word, word_indices):
    print('Saving to pickle')
    with open(pickle_file, "wb") as f:
        pickle.dump([words, indices_word, word_indices], f)
    with open(numpy_file, "wb") as f:
        np.save(f, network_input)
        np.save(f, network_output)

def prepare_data_from_txt_dir(dir, step=3, maxlen=50, vectorization=True, reload_fresh=False):
    if not reload_fresh and os.path.isfile(pickle_file):
        print('Loading from pickle file...')
        return load_from_pickle()

    print('Loading from txt files...')
    svi = []

    for file in glob.glob(dir + "/*.txt"):
        with open(file, 'r', encoding='utf-8') as f:
            temp = f.read()
            svi.append(temp)

    # Combine all text files into one string
    svi = ' '.join(svi)
    # Clean the text: replace newlines with spaces, reduce multiple spaces to single spaces
    svi = re.sub(r'\n', ' ', svi)
    svi = re.sub(r' +', ' ', svi)

    # Tokenize into words (split on whitespace)
    word_list = svi.split()
    # Create a vocabulary of unique words
    words = sorted(set(word_list))

    print('Total unique words:', len(words))
    # Create mappings: word to index and index to word
    word_indices = {word: idx for idx, word in enumerate(words)}
    indices_word = {idx: word for idx, word in enumerate(words)}

    # Convert the text into a list of word indices
    text_as_indices = [word_indices[word] for word in word_list]

    network_input = None
    network_output = None

    if vectorization:
        print('Vectorization...')
        input_sequences = []
        next_words = []

        # Create sequences of length maxlen, with the next word as the target
        for i in range(0, len(text_as_indices) - maxlen, step):
            input_sequences.append(text_as_indices[i:i + maxlen])
            next_words.append(text_as_indices[i + maxlen])
            if i % 10000 == 0:
                print('Sequenced', i, 'of', len(text_as_indices))

        print('Number of sequences:', len(input_sequences))

        # Convert to numpy arrays
        network_input = np.array(input_sequences, dtype=np.int64)
        network_output = np.array(next_words, dtype=np.int64).reshape(-1, 1)

    save_to_pickle(network_input, network_output, words, indices_word, word_indices)
    return network_input, network_output, words, indices_word, word_indices