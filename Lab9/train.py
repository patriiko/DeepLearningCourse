import os
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
import load_dataset
from training_dataset import TrainingDataset

def train():
    text_files_dir = "text_files"

    batch_size = 1000
    model_ver = 'Model_GRU_WordLevel_2'
    epochs = 100
    maxlen = 50

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    print('Torch device:', device)

    # Load the dataset (word-level)
    network_input, network_output, words, indices_word, word_indices = load_dataset.prepare_data_from_txt_dir(
        text_files_dir, maxlen=maxlen)

    writer = SummaryWriter('runs/' + model_ver)

    # Initialize the model
    model = models.MojModel(vocab_size=len(words))
    
    if cuda:
        model = model.to(device)

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())

    # Prepare the dataset and dataloader
    training_dataset = TrainingDataset(network_input, network_output)
    dataloader = DataLoader(training_dataset, batch_size=batch_size, num_workers=0, 
                           shuffle=True, drop_last=True, pin_memory=cuda)

    train_per_epoch = len(dataloader)

    # Training loop
    for epoch in range(epochs):
        model.train()
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
        for i, (input, output) in loop:
            input = input.type(torch.LongTensor).to(device)  # Shape: (batch_size, maxlen)
            output = output.type(torch.LongTensor).to(device).squeeze(-1)  # Shape: (batch_size,)

            optimizer.zero_grad()
            pred = model(input)  # Shape: (batch_size, vocab_size)
            loss = loss_fn(pred, output)
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(loss=loss.item())
            writer.add_scalar('training loss', loss.item(), (epoch * train_per_epoch) + i)

        # Save the model
        path_str = os.path.join('models/', model_ver + '_epoch_' + str(epoch + 1) + '.pth')
        torch.save(model.state_dict(), path_str)

if __name__ == '__main__':
    train()