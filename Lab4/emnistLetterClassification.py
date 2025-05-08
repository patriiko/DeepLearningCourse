from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    
    batch_size = 128
    lr = 0.01
    epochs = 50
    patience = 10
    input_size = 28 * 28
    hidden_size_0 = 512
    hidden_size_1 = 256
    hidden_size_2 = 128
    hidden_size_3 = 64
    hidden_size_4 = 32
    output_size = 26

    writer = SummaryWriter(f'runs/EMNIST/{datetime.now():%Y%m%d-%H%M%S}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds = datasets.EMNIST(root='emnist_data', split='letters',
                               train=True, download=True, transform=transform)
    test_ds  = datasets.EMNIST(root='emnist_data', split='letters',
                               train=False,download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, pin_memory=cuda, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, pin_memory=cuda, num_workers=2)

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size_0),
        nn.ReLU(),
        nn.Linear(hidden_size_0, hidden_size_1),
        nn.ReLU(),
        nn.Linear(hidden_size_1, hidden_size_2),
        nn.ReLU(),
        nn.Linear(hidden_size_2, hidden_size_3),
        nn.ReLU(),
        nn.Linear(hidden_size_3, hidden_size_4),
        nn.ReLU(),
        nn.Linear(hidden_size_4, output_size),
        nn.LogSoftmax(dim=1)
    ).to(device)

    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_acc = 0.0
    epochs_no_improve = 0

    steps_per_epoch = len(train_loader)
    for e in range(1, epochs+1):
        # — Training —
        model.train()
        loop = tqdm(enumerate(train_loader), total=steps_per_epoch, leave=False)
        for idx, (images, labels) in loop:
            images = images.to(device).view(images.size(0), -1)
            labels = (labels-1).to(device)

            optimizer.zero_grad()
            out = model(images)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

            global_step = (e-1)*steps_per_epoch + idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            acc = (out.argmax(1)==labels).float().mean().item()
            writer.add_scalar('Train/Acc', acc, global_step)

            loop.set_description(f"Epoch {e}/{epochs}")
            loop.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc:.3f}")

        # — Evaluacija nakon epoch-e —
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device).view(images.size(0), -1)
                labels = (labels-1).to(device)
                preds = model(images).argmax(1)
                correct += (preds==labels).sum().item()
                total += labels.size(0)
        test_acc = correct/total
        writer.add_scalar('Test/Accuracy', test_acc, e)
        print(f"Epoch {e} → Test Accuracy: {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_emnist_model.pth')
            print(f"  New best! Model saved with acc {best_acc*100:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {e} epochs without improvement.")
                break

    writer.close()
    print(f"\nBest Test Accuracy achieved: {best_acc*100:.2f}%")