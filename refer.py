import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is', device)

def prep_dataset(path='C:\\Users\\abbyx\\Documents\\dataset-ownership-verification_exp\\code\\MNISTimages', train=True, test1=False, test2=False, batch_size=128):
    # split_CIFAR100_to_3('C:\\Users\\abbyx\\Documents\\dataset-ownership-verification_exp\\code')
    trainset, testset1, testset2 = read_mnist_dataset(path, train, test1, test2)

    # train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(trainset, batch_size=batch_size)
    test1_loader = DataLoader(testset1, batch_size=batch_size)
    test2_loader = DataLoader(testset2, batch_size=batch_size, shuffle=True)
    # test2_loader = DataLoader(testset2, batch_size=batch_size)
    
    return train_loader, test1_loader, test2_loader, trainset, testset1, testset2

def train(train_loader, batch_size, net, net_name, test1_loader, testset1, model_name, epoch_num=60):
    # Train
    print('---------- start training ----------')
    net = net # model
    net_name = net_name
    net.to(device)
    net.train()
    learning_rate = 0.001 # low for MNIST
    train_num = len(train_loader)//batch_size
    los = []
    cor = []
    train_los = []
    train_cor = []
    net_corr, net_los, net_train_los, net_train_corr, net_lr, net_epoch = 0, 0, 0, 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epoch_num):
        loss_avg = 0
        train_time = 0
        correct = 0
        num_img = 0
        current_loader = train_loader

        for data in current_loader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            net.train()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.to(device)
            opt.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 20)
            opt.step()
            train_time += 1
            loss_avg += loss.item()*len(labels)
            predict = torch.max(outputs.data, 1)[1]
            correct += (predict == labels).sum()
            num_img += len(labels)
            # print('\n',end='')
            # if train_time % 300 == 0:
            #     print('进度：{}批次'.format(train_time))
            sys.stdout.flush()

        print('\r', end="")


        print('正在训练：{}/{}轮，学习率为：{:.10f}，平均Loss：{:.2f}，正确率为：{:.2f}%'
            .format(epoch+1, epoch_num, opt.state_dict()['param_groups'][0]['lr'], loss_avg/num_img, correct/num_img*100))


        train_cor.append(correct/num_img*100)
        train_los.append(loss_avg/num_img)

        torch.save(net, './'+net_name+'_models/'+model_name+'_model.pkl')
        sys.stdout.flush()

    print('---------- training finished ----------')

    x_epoch = [i for i in range(epoch_num)]

    plt.figure()
    plt.plot(x_epoch, train_los, 'darkorange')
    # plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train_loss'])


    plt.figure()
    plt.plot(x_epoch, torch.tensor(train_cor).cpu())
    plt.xlabel('Epoch')
    plt.ylabel('Correct')
    plt.legend(['Train_Correct'])
    plt.show()

batch_size=4
net = resnet18()
epoch_num = 40
net_name = 'resnet18'
model_name='mnist_aug_test2'
train_loader = prep_dataset()
train(test2_loader, batch_size, net, net_name, train_loader, trainset, model_name, epoch_num)
print('model trained is:', net_name, model_name)