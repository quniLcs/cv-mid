import torch
from torch import nn
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter

from load import load
from util import initialize, optimize, evaluate, save_status

if __name__ == "__main__":
    num_classes = 100
    num_epoch = 30
    batch_size = 256

    train_loader = load(train = True, batch_size = batch_size, shuffle = True)
    test_loader = load(train = False, batch_size = batch_size, shuffle = False)

    print('number of iterations:', num_epoch * len(train_loader))
    print('number of iterations per epoch:', len(train_loader))
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes = num_classes).to(device)
    initialize(model)

    lr = 0.1
    momentum = 0.9
    lambd = 0.006

    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = lambd)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter('../runs/baseline')

    for ind_epoch in range(num_epoch):
        optimize(model, criterion, optimizer, train_loader, lr, ind_epoch, device)

        train_acc, train_loss = evaluate(model, criterion, train_loader, device)
        test_acc, test_loss = evaluate(model, criterion, test_loader, device)

        writer.add_scalars('accuracy', {'train': train_acc, 'test': test_acc}, ind_epoch)
        writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, ind_epoch)

        print('Epoch: %d\tTraining accuracy: %.5f\tTesting Accuracy: %.5f' % (ind_epoch + 1, train_acc, test_acc))

    save_status(model, optimizer, '../result/baseline.pth')

    writer.flush()
    writer.close()