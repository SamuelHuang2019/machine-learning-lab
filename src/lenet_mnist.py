import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

lr = 0.01
momentum = 0.5
log_interval = 10
epochs = 10
batch_size = 64
test_batch_size = 1000


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2 to ensure the same size for input and output
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # define forward propagating process
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # the input and output of nn.Linear() both has dimension one, hence flatten the tensor
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = self.fc4(x)
        return x  # F.softmax(x, dim=1)


# define the details of training for each epoch
def train(epoch):
    # as training mode
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        # convert data in to Variable
        data, target = Variable(data), Variable(target)
        # optimizer default as 0
        optimizer.zero_grad()
        # input data into the network, obtain output
        output = model(data)
        # cross entropy loss function
        loss = F.cross_entropy(output, target)
        # backward propagation gradient
        loss.backward()
        # end last forward and backward propagating，update parameters
        optimizer.step()
        if batch_idx % log_interval == 0:  # 准备打印相关信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test():
    # set test mode
    model.eval()
    # initialize test loss as 0
    test_loss = 0
    # initialize the number of correctly predicted data as 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        # cast to Variable type, hence has gradient
        data, target = Variable(data), Variable(target)

        output = model(data)
        # sum up batch loss
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加

    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # enable GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load training data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True)
    # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
    model = LeNet()  # 实例化一个网络对象
    model = model.to(device)
    # initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # loop through epochs
    for epoch in range(1, epochs + 1):
        train(epoch)
        test()

    # save model
    torch.save(model, '../model.pth')


def custom_test():
    net = torch.load('model.pth')
    print(net)
