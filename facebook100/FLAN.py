import torch.nn as nn
import torch.nn.functional as F
import torch
from gcn.layers import GraphConvolution, MLPLayer
from torch_geometric.utils import one_hot, spmm

class FLAN():
    def __init__(self, args, distance, d=0.1, LPA_step=1,epoch=10):
        super(FLAN, self).__init__()
        self.LPA_step = LPA_step
        self.epoch = epoch
        self.distance = distance
        self.Y_hat = None
        self.X = None
        self.I = torch.eye(args.c).to(args.device)
        self.A_align = None
        self.edge_rate = args.edge_rate2
        self.mask = args.mask
        self.d = d
        self.args = args



    def fit(self, X, Y):
        with torch.no_grad():
            self.Y_hat = Y
            self.X = X
            for i in range(self.epoch):
                for j in range(self.LPA_step):
                    Y_hat2 = self.embed(X, Y, self.Y_hat)
                    # Loss1 = torch.norm(self.embed(Y, X, X) - Y_hat2)

                    self.Y_hat = (1-self.d)*Y_hat2 + self.d*Y
                    self.Y_hat = F.normalize(self.Y_hat)

                    # Loss2 = torch.norm(self.embed(Y, X, X) - self.Y_hat)
                    # print("Loss1:",Loss1, "Loss2:", Loss2)

                    self.Y_hat[self.mask] = Y[self.mask]
                    if i  != self.epoch-1:
                        self.Y_hat[torch.abs(self.Y_hat) < 0.2] = 0  # 0.4
                        self.Y_hat = torch.clip(self.Y_hat, 0.0, 1)
                Y = self.Y_hat


    def embed(self, X, Y, Y_hat):
        coe2 = 1.0 / self.args.beta
        res = torch.mm(torch.transpose(X, 0, 1), X)  # H.T* H
        res2 = torch.mm(torch.transpose(X, 0, 1), Y_hat)  # H.T* Y
        inv = torch.inverse(self.I + coe2 * res)  #
        res3 = torch.mm(inv, res2)  #
        B = coe2 * Y - coe2 * coe2 * torch.mm(X, res3)  # B
        tmp = torch.mm(torch.transpose(Y, 0, 1), B)  # Y.T * B
        part1 = torch.mm(X, tmp)
        part2 = (- self.args.alpha / 2) * (self.distance@B) # torch.mm(distance, B)

        Y_hat = part1 + part2
        return  Y_hat

    def get_X(self):
        # for i in range(self.epoch):
        #     for j in range(self.LPA_step):
        #         X_hat2 = self.embed(self.Y_hat, self.X, self.X)
        #
        #         self.X = (1-self.d)*X_hat2 + self.d*self.X
        #         self.X = F.normalize(self.X)

        # return  self.X
        X_hat = self.embed(self.Y_hat, self.X, self.X)
        res = (1-self.d)*X_hat + self.d*self.X
        res = self.row_l1_normalize(res)

        return res

    def get_aligned_graph(self):
        if self.A_align == None:
            with torch.no_grad():
                coe2 = 1.0 / self.args.beta
                res = torch.mm(torch.transpose(self.X, 0, 1), self.X)  # H.T* H
                inv = torch.inverse(self.I + coe2 * res)
                res2 = torch.mm(torch.transpose(self.Y_hat, 0, 1), self.X) # Y.T*H
                res3 = torch.mm(res2, inv)
                res4 = torch.mm(self.X, res3)
                res5 = coe2 * coe2 * torch.mm(res4,torch.transpose(self.X, 0, 1))
                part1 = coe2 *torch.mm(self.X, torch.transpose(self.Y_hat, 0, 1)) - res5

                res7 = torch.mm(self.X, inv) # H*逆矩阵
                res8 = coe2*torch.eye(self.X.shape[0]).to(self.args.device) - coe2*coe2*torch.mm(res7, torch.transpose(self.X, 0, 1))
                part2 = (- self.args.alpha / 2)*(self.distance @ res8)


                self.A_align = part1 + part2

        return self.A_align

    def get_Y_hat(self):
        return self.Y_hat

    def get_sparse_A(self): # 考虑到有负值，所以将接近0的给mask掉
        kthvalue = torch.kthvalue(
            torch.abs(self.A_align).view(self.A_align.shape[0] * self.A_align.shape[1], 1).T,
            int(self.A_align.shape[0] * self.A_align.shape[0] * self.edge_rate))[0]
        mask = (torch.abs(self.A_align) > kthvalue).detach().float()
        sparse_A = (self.A_align * mask)
        return sparse_A

    def pairwise_distance(self, x, y=None):
        x = x.unsqueeze(0).permute(0, 2, 1)
        if y is None:
            y = x
        y = y.permute(0, 2, 1)  # [B, N, f]
        A = -2 * torch.bmm(y, x)  # [B, N, N]
        A += torch.sum(y ** 2, dim=2, keepdim=True)  # [B, N, 1]
        A += torch.sum(x ** 2, dim=1, keepdim=True)  # [B, 1, N]
        return A.squeeze()

    def row_l1_normalize(self, X):
        norm = 1e-6 + X.sum(dim=1, keepdim=True)
        return X / norm


# Share weight
class FLAN_GCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, tau=0.0):
        super(FLAN_GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.gamma = args.gamma
        self.fc = nn.Linear(nclass, nclass)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc(z))
        return z

    def forward(self, x, adj, adj2):
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = F.relu(self.gc1(x, adj))
        x2 = F.relu(self.gc1(x, adj2))

        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.dropout(x2, self.dropout, training=self.training)

        x1 = self.gc2(x1, adj)
        x2 = self.gc2(x2, adj) # adj

        x = (1 - self.gamma)*x1 + self.gamma*x2
        return x, self.projection(x1), self.projection(x2)


class FLAN_GCN2(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, tau=0.0):
        super(FLAN_GCN2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.gc1_1 = GraphConvolution(nfeat, nhid)
        self.gc2_1 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.gamma = args.gamma

    def forward(self, x, adj, adj2):
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = self.gc2(x1, adj)

        x2 = F.relu(self.gc1_1(x, adj2))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = self.gc2_1(x2, adj2)

        x = (1 - self.gamma)*x1 + self.gamma*x2
        return x

class FLAN_GCN3(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, tau=0.0):
        super(FLAN_GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.gamma = args.gamma
        self.fc = nn.Linear(nclass, nclass)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc(z))
        return z

    def forward(self, x, adj, adj2):
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = F.relu(self.gc1(x, adj))
        x2 = F.relu(self.gc1(x, adj2))

        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.dropout(x2, self.dropout, training=self.training)

        x1 = self.gc2(x1, adj)
        x2 = self.gc2(x2, adj2)

        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.dropout(x2, self.dropout, training=self.training)

        x1 = self.gc3(x1, adj)
        x2 = self.gc3(x2, adj2)

        x = (1 - self.gamma)*x1 + self.gamma*x2
        return x, self.projection(x1), self.projection(x2)