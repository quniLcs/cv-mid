import argparse

import torch
from torch import nn
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter

from load import load
from util import initialize, optimize, evaluate, save_status

if __name__ == "__main__":
    # python main.py --mode baseline
    # python main.py --mode cutout
    # python main.py --mode mixup
    # python main.py --mode cutmix

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", default = 30, type = int)
    parser.add_argument("--batch_size", default = 128, type = int)
    parser.add_argument("--lr", default = 0.1, type = float)
    parser.add_argument("--momentum", default = 0.9, type = float)
    parser.add_argument("--lambd", default = 0.005, type = float)
    parser.add_argument("--mode", default = 'baseline', type = str)
    args = parser.parse_args()

    train_loader = load(train = True, batch_size = args.batch_size, shuffle = True)
    test_loader = load(train = False, batch_size = args.batch_size, shuffle = False)

    print('number of iterations:', args.num_epoch * len(train_loader))
    print('number of iterations per epoch:', len(train_loader))
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes = 100).to(device)
    initialize(model)

    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.lambd)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(args.mode)

    for ind_epoch in range(args.num_epoch):
        optimize(model, criterion, optimizer, train_loader, args.lr, ind_epoch, args.mode, device)

        train_acc, train_loss = evaluate(model, criterion, train_loader, device)
        test_acc, test_loss = evaluate(model, criterion, test_loader, device)

        writer.add_scalars('accuracy', {'train': train_acc, 'test': test_acc}, ind_epoch)
        writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, ind_epoch)

        print('Epoch: %d\tTraining accuracy: %.5f\tTesting Accuracy: %.5f' % (ind_epoch + 1, train_acc, test_acc))

    save_status(model, optimizer, args.mode + '.pth')

    writer.flush()
    writer.close()