from __future__ import division
from __future__ import print_function

import sys
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
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many epochs to wait before logging training status')
parser.add_argument('--gpu-id', type=int, default=0, metavar='N',
                    help='select gpu id (default : 0)')
parser.add_argument('--no-plot', default=True,
                    help='draw plot (default : True)')
parser.add_argument('--num-hidden-units', type=int , default=32, metavar='N',
                    help='how many hidden units (default : 32)')

model_fname = 'nn_model'
stats_fname = 'nn_stats.npz'

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


train, valid, test = LoadData('./toronto_face.npz')

train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)
valid_loader = data_utils.DataLoader(valid, batch_size=args.batch_size, shuffle=True)
test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False)


if args.num_hidden_units == 32:
    model = Net()
    hidden =""
else:
    hidden ="_num_hidden_{}".format(args.num_hidden_units)
    model = custNet(args.num_hidden_units)()

    
if args.cuda:
    torch.cuda.set_device(args.gpu_id)
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

def evaluate(epoch,data_type):
    model.eval()
    valid_loss = 0
    correct = 0
    if data_type == 'train':
        loader = train_loader
    elif data_type == 'validation':
        loader = valid_loader
    elif data_type == 'test':
        loader = test_loader
    
    for data, target in loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data.float(), volatile=True), Variable(target)
        y_pred =model(data)
        valid_loss += F.cross_entropy(y_pred, target).data[0]
        pred = y_pred.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    valid_loss = valid_loss
    valid_loss /= len(loader) # loss function already averages over batch size
    if epoch % args.log_interval == 0 or data_type == 'test':
         print(('Epoch {:3d} {} CE {:.5f} {} Acc {:.5f}').format(epoch, data_type, valid_loss, data_type, correct / len(loader.dataset)))

    return valid_loss, correct / len(loader.dataset)
 
train_ce_list = []
valid_ce_list = []
train_acc_list = []
valid_acc_list = []
for epoch in range(1, args.epochs + 1):
    train(epoch)
    if epoch % 200 == 0:
        SaveModel(('models/MNN_lr{:.3f}_momentum{:.1f}_minibatch{}_epoch{}'+hidden+'.pth').format(args.lr,args.momentum,args.batch_size,epoch),model)
    train_ce, train_acc = evaluate(epoch,'train')
    valid_ce, valid_acc = evaluate(epoch,'validation')
    train_ce_list.append((epoch, train_ce))
    train_acc_list.append((epoch, train_acc))
    valid_ce_list.append((epoch, valid_ce))
    valid_acc_list.append((epoch, valid_acc))

test_ce, test_acc = evaluate(args.epochs,'test')

if epoch % args.no_plot:
    DisplayPlot(train_ce_list, valid_ce_list, 'Cross Entropy', number=0)
    DisplayPlot(train_acc_list, valid_acc_list, 'Accuracy', number=1)



stats = {
    'train_ce': train_ce_list,
    'valid_ce': valid_ce_list,
    'test_ce': test_ce,
    'train_acc': train_acc_list,
    'valid_acc': valid_acc_list,
    'test_acc': test_acc,
}

SaveStats(('stats/MNN_lr{:.3f}_momentum{:.1f}_minibatch{}_epoch{}'+hidden+'.npz').format(args.lr,args.momentum,args.batch_size,args.epochs), stats)