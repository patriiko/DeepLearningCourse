import os
import random
import torch

import models
import load_dataset

text_files_dir = "text_files"
model_to_load = "models/Model_GRU_WordLevel_2_epoch_35.pth"

batch_size = 1000
model_ver = 'Model_GRU_WordLevel'
epochs = 100
maxlen = 50

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

print('Torch version:', torch.__version__)

# Load the dataset (word-level)
network_input, network_output, words, indices_word, word_indices = load_dataset.prepare_data_from_txt_dir(
    text_files_dir, maxlen=maxlen)

# Initialize and load the model
model = models.MojModel(vocab_size=len(words))
model.load_state_dict(torch.load(model_to_load))
model = model.to(device)
model.eval()

# Function to sample the next word
def sample(preds, temperature=1.0):
    preds = preds / temperature
    exp_preds = torch.exp(preds)
    preds = exp_preds / torch.sum(exp_preds)
    probas = torch.multinomial(preds, 1)
    return probas

# Generate text
initial_seed = random.randint(0, len(network_input) - 1)
rolling_values = torch.from_numpy(network_input[initial_seed]).unsqueeze(0).to(device)

# Print the initial sequence
initial_words = [indices_word[idx] for idx in network_input[initial_seed]]
print("Initial sequence:", " ".join(initial_words))

# Generate 1000 words
with torch.no_grad():
    for i in range(1000):
        output = model(rolling_values)  # Shape: (1, radix_size)
        output = sample(output, temperature=0.7)  # Sample the next word
        next_word_idx = output.item()
        next_word = indices_word[next_word_idx]
        
        # Update rolling values (shift left and append new word)
        rolling_values = torch.cat((rolling_values[:, 1:], torch.tensor([[next_word_idx]], device=device)), dim=-1)
        
        # Print the generated word
        print(next_word, end=' ')