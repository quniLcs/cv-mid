import torch
from torch import nn


def initialize(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def descent_lr(lr, optimizer, ind_epoch):
    lr = lr * 2 ** (-ind_epoch / 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def optimize(model, criterion, optimizer, train_loader, lr, ind_epoch, device):
    model.train()

    descent_lr(lr, optimizer, ind_epoch)

    for data in train_loader:
        inputs = data[0].to(device)
        targets = data[1].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, criterion, data_loader, device):
    model.eval()

    count = 0
    losses = 0
    correct = 0

    for data in data_loader:
        inputs = data[0].to(device)
        targets = data[1].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        count += inputs.shape[0]
        _, pred = outputs.max(dim = 1)
        correct += pred.eq(targets).sum().item()
        losses += loss.item() * count

    return correct / count, losses / count


def save_status(model, optimizer, path):
    save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(save_dict, path)


def load_status(model, optimizer, path):
    load_dict = torch.load(path)
    model.load_state_dict(load_dict['model'])
    optimizer.load_state_dict(load_dict['optimizer'])