from __future__ import print_function
import argparse
import random
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import utils
from torch_geometric.data import Data, Dataset,DataLoader
from torch_scatter import scatter_mean
import torch_geometric.transforms as GT
import math
import json
import logging
from model import Net
from dataset import ScitsrDataset

logging.getLogger().setLevel(logging.DEBUG)
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default="/home/jiuxiao/NLP/table/data/SciTSR/train",
                        help='input Train data Path')
    parser.add_argument('--dev_data', type=str, default="/home/jiuxiao/NLP/table/data/SciTSR/dev",
                        help='input Dev data Path')
    parser.add_argument('--test_data', type=str, default="/home/jiuxiao/NLP/table/data/SciTSR/test",
                        help='input Test data Path')
    parser.add_argument('--model_path', default="/home/jiuxiao/NLP/table/GFTSR/GFTSR-pos-img/model", help='Where to store samples and models')
    parser.add_argument('--num_class', type=int, default=3, help='input num_class')
    parser.add_argument('--num_node_features', type=int, default=8, help='input num_node_features')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=32, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
    parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
    parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz\'')
    parser.add_argument('--displayInterval', type=int, default=20, help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=100, help='Number of samples to display when test')
    parser.add_argument('--valInterval', type=int, default=1, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=10, help='Interval to be displayed')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--random_sample', action='store_true',
                        help='whether to sample the dataset with random sampler')
    parser.add_argument('--manualSeed', type=int, default=5, help='Random Seed')
    opt = parser.parse_args()
    return opt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def main(config):
    if config.model_path is None:
        config.model_path = 'model'
    os.system('mkdir {0}'.format(config.model_path))

    config.manualSeed = random.randint(1, 10000)  # fix seed
    logging.info("Random Seed: %f"%(config.manualSeed))
    random.seed(config.manualSeed)
    np.random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)

    train_dataset = ScitsrDataset(config.train_data)
    dev_dataset = ScitsrDataset(config.dev_data)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cpu")
    model = Net(config.num_node_features, config.num_class)
    model.cuda()

    model.apply(weights_init)
    # criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()
    if config.cuda:
        model.cuda()
        criterion = criterion.cuda()

    # loss averager
    loss_avg = utils.averager()
    optimizer = optim.Adam(model.parameters(), lr=config.lr,
                           betas=(config.beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)

    def evaluate(net, dataset, criterion, max_iter=100):
        for p in net.parameters():
            p.requires_grad = False
        net.eval()
        data_loader = DataLoader(dataset, batch_size=config.batch_size)
        val_iter = iter(data_loader)
        i = 0
        n_correct = 0
        n_total = 0
        table_correct = 0
        table_total = 0
        loss_avg = utils.averager()
        # max_iter = min(max_iter, len(data_loader))
        max_iter = len(data_loader)
        for i in range(max_iter):
            data = val_iter.next()
            i += 1
            out_pred = net(data)
            loss = criterion(out_pred, data.y.cuda())
            loss_avg.add(loss)

            _, out_pred = out_pred.max(1)
            label = data.y.detach().cpu().numpy()
            out_pred = out_pred.detach().cpu().numpy()
            if (label == out_pred).all():
                table_correct = table_correct + 1
            table_total = table_total + 1
            n_correct = n_correct + (label == out_pred).sum()
            n_total = n_total + label.shape[0]
            # print("correct:",n_correct,label.shape[0])
        accuracy = n_correct / float(n_total)
        table_accuracy = table_correct / float(table_total)

        logging.info('Test cell loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
        logging.info('Test one table loss: %f, accuray: %f' % (loss_avg.val(), table_accuracy))
        return loss_avg

    def train():
        for epoch in range(config.num_epoch):
            train_iter = iter(train_loader)
            i = 0
            while i < len(train_loader):
                for p in model.parameters():
                    p.requires_grad = True
                model.train()

                data = train_iter.next()
                out_pred = model(data)
                loss = criterion(out_pred, data.y.cuda())
                model.zero_grad()
                loss.backward()
                optimizer.step()

                loss_avg.add(loss)

                i += 1
                if i % config.displayInterval == 0:
                    logging.info('[%d/%d][%d/%d] Loss: %f' %
                          (epoch, config.num_epoch, i, len(train_loader), loss_avg.val()))
                    loss_avg.reset()

            if epoch % config.valInterval == 0:
                loss_val = evaluate(model, dev_dataset, criterion)
                scheduler.step(loss_val.val())

            if epoch % config.saveInterval == 0:
                torch.save(model.state_dict(), '{0}/net_{1}_{2}.pth'.format(config.model_path, epoch, i))
                # for k,v in crnn.state_dict().items():
                #     print(k)
    train()

if __name__ == "__main__":
    args = config()
    main(args)