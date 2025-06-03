from torch.utils.data import Dataset

class TrainingDataset(Dataset):
    def __init__(self, network_input, network_output):
        self.network_input = network_input
        self.network_output = network_output

    def __len__(self):
        return len(self.network_input)

    def __getitem__(self, idx):
        return self.network_input[idx], self.network_output[idx]