import torch
from torch import nn

from cutout import cutout
from mixup import mixup
from cutmix import cutmix


def initialize(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def descent_lr(lr, ind_epoch, optimizer):
    lr = lr * 2 ** (-ind_epoch / 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def optimize(model, criterion, optimizer, train_loader, lr, ind_epoch, mode, device):
    model.train()
    descent_lr(lr, ind_epoch, optimizer)

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if mode == 'baseline':
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        elif mode == 'cutout':
            cutout(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        elif mode == 'mixup':
            inputs_mixup, targets_mixup, lambd, index = mixup(inputs, targets)
            outputs = model(inputs_mixup)
            loss = lambd * criterion(outputs, targets) + (1 - lambd) * criterion(outputs, targets_mixup)
        else:  # mode == 'cutmix'
            inputs_cutmix, targets_cutmix, lambd, index = cutmix(inputs, targets)
            outputs = model(inputs_cutmix)
            loss = lambd * criterion(outputs, targets) + (1 - lambd) * criterion(outputs, targets_cutmix)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, criterion, data_loader, device):
    model.eval()

    count = 0
    losses = 0
    correct = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

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