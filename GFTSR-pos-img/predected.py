import os
import argparse
import torch
import logging

from torch_geometric.data import Data, Dataset,DataLoader
from model import Net
from dataset import ScitsrDataset

logging.getLogger().setLevel(logging.DEBUG)
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default="/home/jiuxiao/NLP/table/data/SciTSR/train",
                        help='input Train data Path')
    parser.add_argument('--dev_data', type=str, default="/home/jiuxiao/NLP/table/data/SciTSR/dev",
                        help='input Dev data Path')
    parser.add_argument('--test_data', type=str, default="/home/jiuxiao/NLP/table/data/predectData/test",
                        help='input Test data Path')
    parser.add_argument('--model_path', default="/home/jiuxiao/NLP/table/GFTE/GFTE-pos/model/net_18_361.pth",
                        help='Where to store samples and models')
    parser.add_argument('--num_class', type=int, default=3, help='input num_class')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--num_node_features', type=int, default=8, help='input num_node_features')
    opt = parser.parse_args()
    return opt

def predect(config):

    test_dataset = ScitsrDataset(config.dev_data)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = Net(config.num_node_features, config.num_class)
    model.cuda()
    model.load_state_dict(torch.load(config.model_path))

    val_iter = iter(test_loader)
    # data = val_iter.next()
    n_correct = 0
    n_total = 0
    for data in val_iter:
        out_pred = model(data)
        _, pred = out_pred.max(dim=1)
        label = data.y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        epoch_ped = (label == pred).sum()
        if (label == pred).all():
            n_correct = n_correct + 1
        n_total = n_total + 1
        logging.info("len_label = %d, epoch_ped = %d, n_correct = %d, n_total = %d"%
                     (len(label), epoch_ped, n_correct, n_total))
    accuracy = n_correct / float(n_total)
    logging.info('Test accuray: %f' % (accuracy))



if __name__ == "__main__":
    print("predect text:")
    opt = config()
    predect(opt)
