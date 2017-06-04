from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from util import *
from multi_layer_nn import *


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=34, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


train, valid, test = LoadData('./toronto_face.npz')

train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)
valid_loader = data_utils.DataLoader(valid, batch_size=args.batch_size, shuffle=True)
test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data.float()), Variable(target)
        optimizer.zero_grad()
        y_pred =model(data)
        loss = F.cross_entropy(y_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
def valid(epoch):
    model.eval()
    valid_loss = 0
    correct = 0
    for data, target in valid_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data.float(), volatile=True), Variable(target)
        y_pred =model(data)
        valid_loss += F.cross_entropy(y_pred, target).data[0]
        pred = y_pred.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    valid_loss = valid_loss
    valid_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nvalid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
                

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data.float(), volatile=True), Variable(target)
        y_pred =model(data)
        test_loss += F.cross_entropy(y_pred, target).data[0]
        pred = y_pred.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        
for epoch in range(1, args.epochs + 1):
    train(epoch)
    valid(epoch)


test(num_epochs+1)