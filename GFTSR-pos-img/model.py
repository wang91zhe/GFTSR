import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.lin1 = torch.nn.Linear(64*2, 64) #pair 到 单个
        self.lin_img =torch.nn.Linear(64*2, 64)
        self.lin_final = torch.nn.Linear(64*2, num_classes)

        ks = [3, 3, 3]
        ps = [1, 1, 1]
        ss = [1, 1, 1]
        nm = [64, 64, 64]
        cnn = nn.Sequential()
        leakyRelu = False

        def convRelu(i, batchNormalization=False):
            nIn = 1 if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  #
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  #
        convRelu(2, True)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # bs, 64, 32, 32
        self.cnn = cnn

    # 从cnn的结果中，取出每个box中心点对应的feature
    def sample_box_feature(self, cnnout, nodenum, pos):
        '''
        :param cnnout: Bs*channel*H*W
        :param nodenum:  BS 表示每个图有多少节点。batch中有多个图
        :param pos: box的坐标。中心点 （也可以考虑4个端点） (N_nodes*[y,x])
        :return:
        '''
        cnt = 0
        # print("**",nodenum)
        for i in range(nodenum.size()[0]):
            # print("**",pos.size())
            imgpos = pos[cnt:cnt + nodenum[i], :]
            imgpos = imgpos.unsqueeze(0)
            imgpos = imgpos.unsqueeze(0)  # make 1*1*W*2 , W 看作是一个图有多少个box
            # print("**",imgpos.size())
            cnnin = cnnout[i].unsqueeze(0)  # 第0维是batch
            sout = F.grid_sample(cnnin, imgpos, mode='bilinear', padding_mode='border', align_corners=True)
            cnt += nodenum[i]
            sout = sout.squeeze(0)
            sout = sout.squeeze(1)
            sout = sout.permute(1, 0)  # num_box*feature_num
            if i == 0:
                out = sout
            else:
                out = torch.cat((out, sout), 0)
        return out

    def forward(self, data):
        x, edge_index, img, nodenum, imgpos = data.x, data.edge_index, data.img, data.nodenum, data.imgpos
        x = x.cuda()
        edge_index = edge_index.cuda()
        img = img.cuda()
        nodenum = nodenum.cuda()
        imgpos = imgpos.cuda()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x1 = x[edge_index[0]]
        x2 = x[edge_index[1]]
        xpair = torch.cat((x1, x2), dim=1)
        xpair = F.relu(self.lin1(xpair))

        # combine img node feature
        imgconv = self.cnn(img)  # bs, 64, 32, 32
        ximg = self.sample_box_feature(imgconv, nodenum, imgpos)
        x1img = ximg[edge_index[0]]
        x2img = ximg[edge_index[1]]
        ximgpair = torch.cat((x1img, x2img), dim=1)
        ximgpair = F.relu(self.lin_img(ximgpair))

        # final
        xfin = torch.cat((xpair, ximgpair), dim=1)
        xfin = (self.lin_final(xfin))
        return F.log_softmax(xfin, dim=1)

if __name__ == "__main__":  
    print("test")
    