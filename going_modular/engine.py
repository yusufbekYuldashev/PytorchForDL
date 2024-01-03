"""
Contains code for training and testing step and train function.
"""
from typing import Dict, List, Tuple
import torch
from tqdm.auto import tqdm

def train_step(model, dataloader, loss_fn, optim, device):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        train_acc += (y == torch.softmax(pred, dim=1).argmax(dim=1)).sum().item() / len(pred)
        optim.zero_grad()
        loss.backward()
        optim.step()
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            test_acc += (y==torch.softmax(pred, dim=1).argmax(dim=1)).sum().item() / len(pred)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc

def train(model, train_dataloader, test_dataloader, optim, device, loss_fn=torch.nn.CrossEntropyLoss, epochs=5):
    print('Starting training')
    results = {'train_loss': [],
               'train_acc': [],
               'test_loss':[],
               'test_acc':[]}
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}---------------------------------")
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optim, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        print(f"Epoch: {epoch} Train acc: {train_acc:.2f} Train Loss: {train_loss:.4f} Test Acc: {test_acc:.2f} Test Loss: {test_loss:.4f}")
        results['train_loss'].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results
