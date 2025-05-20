import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import os
import numpy as np

def main():
    TRAIN_DIR = 'Brain Tumor MRI Dataset/Training'
    TEST_DIR = 'Brain Tumor MRI Dataset/Testing'
    IMG_SIZE = 224
    BATCH_SIZE = 16
    NUM_EPOCHS = 60
    NUM_CLASSES = 4
    CHECKPOINT_DIR = 'checkpoints'
    LOAD_CHECKPOINT = True

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomRotation(45),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = ImageFolder(TEST_DIR, transform=test_transform)

    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),  # Dodan dropout
        nn.Linear(512, NUM_CLASSES)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    writer = SummaryWriter()

    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 7

    start_epoch = 34
    if LOAD_CHECKPOINT and os.path.exists(CHECKPOINT_DIR):
        checkpoint_path = sorted(os.listdir(CHECKPOINT_DIR))[-1]
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, checkpoint_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Učitavanje checkpointa iz epohe {checkpoint['epoch']}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == labels).sum().item()
            correct_train += batch_correct
            total_train += labels.size(0)
            
            batch_acc = 100 * batch_correct / labels.size(0)
            progress_bar.set_postfix(loss=loss.item(), acc=batch_acc)
            
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Train/Batch_Acc', batch_acc, epoch * len(train_loader) + batch_idx)

        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = 100 * correct_val / total_val
        scheduler.step(val_acc)
        
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        writer.add_scalar('Train/Epoch_Acc', train_acc, epoch)
        writer.add_scalar('Val/Loss', val_loss / len(test_loader), epoch)
        writer.add_scalar('Val/Acc', val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'acc': train_acc,
                'val_acc': val_acc
            }, os.path.join(CHECKPOINT_DIR, f'best_checkpoint.pth'))
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping na epohi {epoch+1}")
                break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'acc': train_acc,
            'val_acc': val_acc
        }, os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'))

    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_checkpoint.pth'))['model_state_dict'])
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluacija'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    print(f'\nFinalna točnost na testnom skupu: {accuracy:.2f}%')
    for i in range(NUM_CLASSES):
        print(f'Točnost za klasu {train_dataset.classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    writer.add_scalar('Accuracy/test', accuracy)
    writer.close()

if __name__ == '__main__':
    main()
