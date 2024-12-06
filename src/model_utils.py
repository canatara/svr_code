# Code adopted from https://github.com/pytorch/examples/tree/main/mnist


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import numpy as np


class Net(nn.Module):
    def __init__(self, load=None, seed=42):

        torch.manual_seed(seed)

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()

        self.max_pool2d = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(9216, 256)
        self.relu_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU()

        self.fc4 = nn.Linear(64, 10)

        if torch.backends.mps.is_built():
            self.device = torch.device("mps")
        elif torch.backends.cuda.is_built():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if load is not None:
            self.load_state_dict(torch.load(load, map_location=self.device))

        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2d(x)
        x = self.dropout1(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu_fc1(x)

        x = self.fc2(x)
        x = self.relu_fc2(x)

        x = self.fc3(x)
        x = self.relu_fc3(x)

        x = self.dropout2(x)
        x = self.fc4(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net2(nn.Module):
    def __init__(self, load=None, seed=42):

        torch.manual_seed(seed)

        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.relu_conv1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.relu_conv2 = nn.ReLU()

        self.zero_pad = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.relu_conv3 = nn.ReLU()

        self.max_pool2d = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(9216, 256)
        self.relu_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU()

        self.fc4 = nn.Linear(64, 10)

        if torch.backends.mps.is_built():
            self.device = torch.device("mps")
        elif torch.backends.cuda.is_built():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if load is not None:
            self.load_state_dict(torch.load(load, map_location=self.device))

        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)

        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.zero_pad(x)

        x = self.conv3(x)
        x = self.relu_conv3(x)

        x = self.max_pool2d(x)
        x = self.dropout1(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu_fc1(x)

        x = self.fc2(x)
        x = self.relu_fc2(x)

        x = self.fc3(x)
        x = self.relu_fc3(x)

        x = self.dropout2(x)
        x = self.fc4(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net3(nn.Module):
    def __init__(self, load=None, seed=42):

        torch.manual_seed(seed)

        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.relu_conv1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.relu_conv2 = nn.ReLU()

        self.zero_pad = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.relu_conv3 = nn.ReLU()

        self.max_pool2d = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(9216, 512)
        self.relu_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.relu_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 128)
        self.relu_fc3 = nn.ReLU()

        self.fc4 = nn.Linear(128, 10)

        if torch.backends.mps.is_built():
            self.device = torch.device("mps")
        elif torch.backends.cuda.is_built():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if load is not None:
            self.load_state_dict(torch.load(load, map_location=self.device))

        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)

        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.zero_pad(x)

        x = self.conv3(x)
        x = self.relu_conv3(x)

        x = self.max_pool2d(x)
        x = self.dropout1(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu_fc1(x)

        x = self.fc2(x)
        x = self.relu_fc2(x)

        x = self.fc3(x)
        x = self.relu_fc3(x)

        x = self.dropout2(x)
        x = self.fc4(x)

        output = F.log_softmax(x, dim=1)
        return output


@torch.no_grad()
def get_model_activations(model, layer, manifold, flatten=True, to_numpy=True, rand_proj_dim=100):

    try:
        device = model.device
    except Exception:
        device = next(model.parameters()).device

    model_dict = dict(model.named_children())

    try:
        assert layer in model_dict.keys(), f"{layer} is not in {model_dict.keys()}"

        modules = []
        for key, val in model_dict.items():
            modules += [val]
            if key == layer:
                break

        # if 'conv' in layer:
        #     # modules += [nn.AdaptiveAvgPool2d(2)]
        #     modules += [nn.AdaptiveAvgPool2d((1, 1))]

        modules = nn.Sequential(*modules)

        assert len(manifold.shape) == 5
        angle_size, manifold_size, ch, h, w = manifold.shape

        act = manifold.reshape(angle_size*manifold_size, ch, h, w)

        k = 1
        is_oom = True
        projected_out = []
        while is_oom:

            try:
                N = len(act)//k
                for i in range(k):
                    start = i*N
                    end = (i+1)*N if i != k-1 else len(act)
                    out = modules(act[start:end].to(device)).cpu()
                    projected_out += [out]
                is_oom = False

            except torch.cuda.OutOfMemoryError:
                k = k + 1
                projected_out = []
                print(f"Cuda OOM - Trying {k} batched random projection")

            except Exception as e:
                raise e

        act = torch.cat(projected_out)

    except Exception as e:

        if layer is None:
            act = manifold
        else:
            raise e

    act = act.to(torch.float64)
    if to_numpy:
        act = act.numpy()

    if flatten:
        act = act.reshape(angle_size, manifold_size, -1)
    else:
        act = act.reshape(angle_size, manifold_size, *act.shape[1:])

    if rand_proj_dim is not None and to_numpy and flatten:
        from sklearn import random_projection
        transformer = random_projection.GaussianRandomProjection(random_state=42, n_components=rand_proj_dim)
        act = transformer.fit_transform(act.reshape(angle_size*manifold_size, -1))
        act = act.reshape(angle_size, manifold_size, -1)

    return act


def trainer(model, args):

    use_cuda = args.use_cuda and torch.cuda.is_available()
    use_mps = args.use_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    aug_transform = []
    if args.augment:
        aug_transform += [transforms.CenterCrop(26),
                          transforms.Resize((28, 28)),
                          transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                          transforms.RandomRotation(180),
                          # transforms.RandomAffine(5)
                          ]
    transform = [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transforms.Compose(aug_transform+transform))
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transforms.Compose(transform))
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    metrics = dict(epoch=[])
    for epoch in range(1, args.epochs + 1):

        metrics['epoch'] += [epoch]
        train(args, model, device, train_loader, optimizer, epoch, metrics)
        test(model, device, test_loader, metrics)
        scheduler.step()

    if args.save_model:
        filename = f"mnist_cnn_lr_{args.lr:.1f}_batch_size_{args.batch_size}_epoch_{args.epochs}.pt"
        if args.augment:
            filename = 'augmented_' + filename

        torch.save(model.state_dict(), filename)

    return metrics


class Trainer:

    def __init__(self, model, args):

        self.model = model
        self.args = args

        model_type = model.__class__.__name__

        self.filename = f"{model_type}_mnist_cnn_lr_{args.lr:.1f}_batch_size"\
            f"_{args.batch_size}_epoch_{args.epochs}_seed_{args.seed}.pt"
        if self.args.augment:
            self.filename = 'augmented_' + self.filename

        use_cuda = args.use_cuda and torch.cuda.is_available()
        use_mps = args.use_mps and torch.backends.mps.is_available()

        torch.manual_seed(args.seed)

        if use_cuda:
            device = torch.device("cuda")
        elif use_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        train_kwargs = {'batch_size': args.batch_size}
        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 8,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        aug_transform = []
        if args.augment:
            aug_transform += [transforms.CenterCrop(26),
                              transforms.Resize((28, 28)),
                              transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                              transforms.RandomRotation(180),
                              # transforms.RandomAffine(5)
                              ]
        transform = [transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))
                     ]

        dataset1 = datasets.MNIST('../data', train=True, download=True,
                                  transform=transforms.Compose(aug_transform+transform))
        dataset2 = datasets.MNIST('../data', train=False,
                                  transform=transforms.Compose(transform))
        self.train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=args.gamma)

    def train(self):

        args = self.args

        metrics = dict(epoch=[])
        for epoch in range(1, args.epochs + 1):

            metrics['epoch'] += [epoch]
            train(args, self.model, self.device, self.train_loader, self.optimizer, epoch, metrics)
            test(self.model, self.device, self.test_loader, metrics)
            self.scheduler.step()

        if args.save_model:
            torch.save(self.model.state_dict(), self.filename)

        return metrics

    def test(self):

        metrics = dict(epoch=[])

        test(self.model, self.device, self.test_loader, metrics)

        return metrics


def train(args, model, device, train_loader, optimizer, epoch, metrics=None):
    model.train()

    batch_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

        batch_loss += [loss.item()]

    if type(metrics) is dict:
        metrics['mean_loss'] = metrics.get('mean_loss', []) + [np.mean(batch_loss)]
        metrics['std_loss'] = metrics.get('std_loss', []) + [np.std(batch_loss)/np.sqrt(len(batch_loss))]


def test(model, device, test_loader, metrics):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if type(metrics) is dict:
        metrics['test_loss'] = metrics.get('test_loss', []) + [test_loss]


def default_parser():

    import argparse

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--augment', action='store_true', default=False,
                        help="Augment dataset or not")
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--use-mps', action='store_true', default=False,
                        help='disables macOS GPU training')

    parser.add_argument('--lr', type=float, default=0.5,
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs to train (default: 14)')

    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action="store_true", default=False,
                        help='Save model')

    return parser
