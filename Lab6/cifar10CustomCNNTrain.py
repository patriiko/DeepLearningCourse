import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

import cifar10CustomCNNModels

def train():
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cpu')
    if cuda:
        device = torch.device('cuda')

    batch_size = 256
    
    writer = SummaryWriter('runs/CIFAR_Net1')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10('cifar_data', download=True, train=True, transform=transform_train)
    testset = datasets.CIFAR10('cifar_data', download=True, train=False, transform=transform_test)


    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = cifar10CustomCNNModels.Net_Cifar().to(device)

    loss_fn = nn.NLLLoss().to(device)
    lrate = 0.0003

    #optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lrate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    epochs = 100
    train_per_epoch = int(len(trainset) / batch_size)
    for e in range(epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for idx, (images, labels) in loop:

            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(images)
            labels = labels.to(device, non_blocking=True)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.item(), (e * train_per_epoch) + idx)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            accuracy = correct / len(predictions)
            loop.set_description(f"Epoch [{e}/{epochs}")
            loop.set_postfix(loss=loss.item(), acc=accuracy)
            writer.add_scalar('acc', accuracy, (e * train_per_epoch) + idx)

        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device=device)
                y = y.to(device=device)

                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            test_acc = float(num_correct) / float(num_samples) * 100

            print(f'Dobio sam točnih {num_correct} od ukupno {num_samples} što čini točnost od {float(num_correct) / float(num_samples) * 100:.2f}%')

        scheduler.step(test_acc)

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Dobio sam točnih {num_correct} od ukupno {num_samples} što čini točnost od {float(num_correct) / float(num_samples) * 100:.2f}%')


if __name__ == '__main__':
    train()