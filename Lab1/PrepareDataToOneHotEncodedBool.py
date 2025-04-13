import glob
import numpy as np
import os.path
import re
import timeit
import torch
import pickle

pickle_file = 'cache_data.pkl'
numpy_file = 'numpy_data.npy'
text_dir = 'txt_files'

def load_from_pickle():
    print('Opening from pickle cache')
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    with open(numpy_file, "rb") as f:
        network_input = np.load(f)
        network_output = np.load(f)
        network_input_labels = np.load(f)
    return network_input, network_output, network_input_labels, data[0], data[1], data[2], data[3], data[4]


def save_to_pickle(network_input, network_output, network_input_labels, words, indices_char, char_indices, label_indices, indices_label):
    print('Saving to pickle')
    with open(pickle_file, "wb") as f:
        pickle.dump([words, indices_char, char_indices, label_indices, indices_label], f)
    with open(numpy_file, "wb") as f:
        np.save(f, network_input)
        np.save(f, network_output)
        np.save(f, network_input_labels)



def prepare_data_from_csv(dir, maxlen, word_size = 1, step=3, vectorization=True, reloadFresh=False):
    if not reloadFresh and os.path.isfile(pickle_file):
        print('Loading from pickle file...')
        return load_from_pickle()

    print('Loading from txt files...')
    input_data = []
    next_chars = []
    tree_labels = []
    svi = []
    dic = {}

    #trees = []
    for file in glob.glob(dir + "**/*.txt"):
        label = file.split('/')[-2] #uzimamo zadnji folder kao label
        if label not in dic:
            dic[label] = []
        with open(file, 'r') as f:
            temp = f.read()
            visak = len(temp) % word_size
            if visak > 0:
                temp = temp[:-visak]
            dic[label].append(temp)
            svi.append(temp)

    svi = ''.join(svi)

    lista = list(map(''.join, zip(*[iter(svi)] * word_size))) #dijelim listu po veličini riječi
    words = sorted(set(lista))

    print('total words:', len(words))
    char_indices = dict((c, i) for i, c in enumerate(words))
    indices_char = dict((i, c) for i, c in enumerate(words))

    label_indices = dict((c, i) for i, c in enumerate(dic.keys()))
    indices_label = dict((i, c) for i, c in enumerate(dic.keys()))


    network_input = None
    network_output = None

    if vectorization: # radimo vektorizaciju po mapi zato što nam trebaju klase

        #prvo presložimo mapu kako bi u svakoj imali string
        for svaki in dic.keys():
            print("Converting label", svaki)
            dic[svaki] = ''.join(dic[svaki])
            current_label = dic[svaki]

            for i in range(0, len(current_label) - maxlen * word_size, step * word_size):
                input_data.append(current_label[i: i + (maxlen * word_size)])
                next_chars.append(current_label[i + (maxlen * word_size): i + (maxlen * word_size) + word_size])
                tree_labels.append(label_indices[svaki])
        print('nb sequences:', len(input_data))

        print('Vectorization...')
        network_input = np.zeros((len(input_data), maxlen, len(words)), dtype=np.bool)
        network_output = np.zeros((len(input_data), len(words)), dtype=np.bool)
        network_input_labels = np.zeros((len(input_data), len(label_indices)), dtype=np.bool)

        for i, item in enumerate(input_data):
            for t, char in enumerate2(item, start=0, step=word_size):
                network_input[i, int(t / word_size), char_indices[char]] = 1
            network_output[i, char_indices[next_chars[i]]] = 1
            network_input_labels[i, tree_labels[i]] = 1
            if i % 10000 == 0:
                print('Vectorized', i, 'of', len(input_data))

    save_to_pickle(network_input, network_output, network_input_labels, words, indices_char, char_indices, label_indices, indices_label)

    return network_input, network_output, network_input_labels, words, indices_char, char_indices, label_indices, indices_label


def enumerate2(xs, start=0, step=1):
    for i in range(0, len(xs), step):
        yield (start, xs[i:i + step])
        start += step


# Učitaj sav tekst iz txt_files/ foldera
sve_knjige_tekst = ""
for input_file in glob.glob(os.path.join(text_dir, "*.txt")):
    with open(input_file, 'r') as f:
        sve_knjige_tekst += f.read()

# Generiraj skup i indekse karaktera
set_karaktera = sorted(set(sve_knjige_tekst))
char_index = {c: i for i, c in enumerate(set_karaktera)}
index_char = {i: c for i, c in enumerate(set_karaktera)}

# Kreiraj podnizove za učenje
seq_len = 50
step = 4
lista_za_ucenje = []

for i in range(0, len(sve_knjige_tekst) - seq_len, step):
    sekvenca = sve_knjige_tekst[i:i + seq_len]
    lista_za_ucenje.append(sekvenca)

# Pretvori u one-hot encoding
n = len(lista_za_ucenje)
m = len(set_karaktera)
network_input = np.zeros((n, seq_len, m), dtype=np.bool_)

for i, sekvenca in enumerate(lista_za_ucenje):
    for t, char in enumerate(sekvenca):
        network_input[i, t, char_index[char]] = 1

# Spremi u NumPy datoteku
with open(numpy_file, "wb") as f:
    np.save(f, network_input)

print("One-hot encoding complete and saved to", numpy_file)