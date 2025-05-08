import os
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from customCNNClassificationModels import RobustCNN  # tvoj model

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 128
    writer = SummaryWriter('runs/Robust_CNN_150x150')

    train_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    train_dir = "seg_train/seg_train"
    test_dir = "seg_test/seg_test"

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = len(os.listdir(train_dir))
    model = RobustCNN(num_classes).to(device)

    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    epochs = 30
    train_per_epoch = len(train_loader)

    for epoch in range(epochs):
        model.train()
        loop = tqdm(enumerate(train_loader), total=train_per_epoch, leave=True)
        for idx, (images, labels) in loop:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train', loss.item(), (epoch * train_per_epoch) + idx)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            accuracy = correct / len(predictions)
            writer.add_scalar('Accuracy/train', accuracy, (epoch * train_per_epoch) + idx)

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item(), acc=accuracy)

    print("✅ Treniranje gotovo.")

    # Evaluacija
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    final_loss = test_loss / len(test_loader)
    final_acc = correct / total

    print(f"📊 Test Loss: {final_loss:.4f}, Test Accuracy: {final_acc:.4f}")

    writer.add_hparams(
        {'lr': 0.01, 'batch_size': batch_size},
        {'hparam/test_accuracy': final_acc, 'hparam/test_loss': final_loss}
    )

    writer.close()

if __name__ == "__main__":
    train()
