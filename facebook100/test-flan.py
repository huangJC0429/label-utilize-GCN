from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import copy
import random
import torch.nn.functional as F
import torch.optim as optim
from torch_sparse import SparseTensor

from utils import load_data, accuracy, get_dataset, row_l1_normalize, GCN_norm, get_dataset_split, pairwise_distance, \
    convert_tensor_to_sparse_tensor, index_to_mask
from FLAN import FLAN, FLAN_GCN, FLAN_GCN2
from tqdm import trange
from datasets import load_dataset

exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10, help='The number of experiments.')

parser.add_argument('--dataset', default='Caltech36', help='Dataset string.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--edge_rate', type=float, default=0.99, help='sparse feature distance matrix.') # 0.9代表mask90%的
parser.add_argument('--edge_rate2', type=float, default=0.9, help='sparse learned align_A.')

parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--I', type=bool, default=False, help='Lamda')


parser.add_argument('--alpha', type=float, default=0, help=' for flan.')
parser.add_argument('--beta', type=float, default=1000.0, help=' for flan.')
parser.add_argument('--gamma', type=float, default=0.1, help=' for combine GCN and FLAN-GCN.')
parser.add_argument('--tau', type=float, default=0.8, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=0.1, help='Lamda')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device
args.cuda = torch.cuda.is_available()

if args.I:
    args.I = 1.0
else:
    args.I = 0.0

adj, features, labels,idx_train,idx_val,idx_test = load_dataset(args)
# print(adj)
# exit()
#
# path = "data"
# # Load data
# # adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)
# dataset, evaluator = get_dataset(
#     name=args.dataset,
#     root_dir=path,
# )
# data = dataset._data.to(device)
# adj, features, idx_train, idx_val, idx_test, labels = data.edge_index, data.x, data.train_mask, data.val_mask, data.test_mask, data.y
# features_normalized = row_l1_normalize(features)
# adj_normalized = GCN_norm(adj, features)

adj_normalized = adj
features_normalized = features

# knn_adj_norm = torch.load(f'./knn_graph/{args.dataset}_sims_{args.k}.pt')


if args.cuda:
    adj_normalized = adj_normalized.to(device)
    labels = labels.squeeze().long().to(device)
    features_normalized = features_normalized.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

# A = torch.tensor([[1.0, 1.0],
#                   [1.0, 2.0]])
# print(F.normalize(A))
# exit()

# feature distance
feature_distance = pairwise_distance(features_normalized)
feature_distance = F.normalize(feature_distance)
kthvalue = torch.kthvalue(
    feature_distance.view(feature_distance.shape[0] * feature_distance.shape[1], 1).T,
    int(feature_distance.shape[0] * feature_distance.shape[1] * args.edge_rate))[0]
mask = (feature_distance > kthvalue).detach().float()
feature_distance = (feature_distance * mask)
# 转化为torch_sparse
feature_distance = convert_tensor_to_sparse_tensor(feature_distance)
# print(feature_distance.size(0))
# exit()


X_MLP = torch.load("./X_MLP/" + args.dataset + "_X_MLP"+".pt")
print(X_MLP.shape)
# exit()

# print(idx_train)
# exit()
idx_train = index_to_mask(idx_train, features_normalized.shape[0])
idx_val = index_to_mask(idx_val, features_normalized.shape[0])
idx_test = index_to_mask(idx_test, features_normalized.shape[0])

# exit()
args.train_mask = idx_train
args.c = max(labels) + 1
# print(args.c)
# exit()
# print(labels[idx_train].shape)
# print(max(labels))
# exit()

args.mask = args.train_mask
aligned_labels = torch.load("./aligned_nodes/"+args.dataset+"_aligned_label.pt")
print(max(aligned_labels))
# exit()
F_mask = torch.load("./UNF_mask/"+args.dataset+"_F_mask.pt")
args.mask = (args.train_mask|F_mask)
aligned_labels = F.one_hot(aligned_labels)
k_lable = torch.zeros_like(aligned_labels)
k_lable[args.mask] = aligned_labels[args.mask]

# aligned_labels = F.one_hot(labels)
# k_lable = torch.zeros_like(aligned_labels)
# k_lable[idx_train] = aligned_labels[idx_train]
# F_mask = torch.load("./UNF_mask/"+args.dataset+"_F_mask.pt")


# x_labels = F.one_hot(labels)
# k_lable = torch.zeros_like(x_labels)
# k_lable[idx_train] = x_labels[idx_train]

# torch.set_printoptions(profile="full")

flan = FLAN(args, feature_distance, d=0.3, LPA_step=1,epoch=200) #0.3, 1, 200
# flan.fit(X_MLP, F.one_hot(k_lable).float())
flan.fit(X_MLP, k_lable.float())

# 获取SX
X_hat = flan.get_X()
# print(X_hat)
# exit()
acc_test = accuracy(flan.Y_hat[idx_test], labels[idx_test])
# print(acc_test)

# 获取邻接矩阵
aligned_A = flan.get_aligned_graph()
# print(aligned_A)
sparse_aligned_A = flan.get_sparse_A().detach()
sparse_aligned_A = convert_tensor_to_sparse_tensor(sparse_aligned_A)
# print(sparse_aligned_A)
# exit()


# Stage 2


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    # return torch.diagonal(torch.mm(z1, z2.t())).mean()
    return torch.sum((z1 - z2)**2, dim=1).mean()

def conresive_loss(X1, X2, output=None, F_MASK=None):
    UNF_MASK = ~F_MASK
    probs = torch.exp(output)
    f = lambda x: torch.exp(x / args.tau)
    pos_sim = f(sim(X1[F_MASK], X2[F_MASK]))
    neg_sim = f(sim(X1[UNF_MASK], X2[UNF_MASK]))

    # The second term in Eq. (10): uniformity loss
    intra_c = (X1).T @ (X1).contiguous()
    intra_c = torch.exp(F.normalize(intra_c, p=2, dim=1)).sum()
    loss_uni = torch.log(intra_c).mean()

    intra_c_2 = (X2).T @ (X2).contiguous()
    intra_c_2 = torch.exp(F.normalize(intra_c_2, p=2, dim=1)).sum()
    loss_uni += torch.log(intra_c_2).mean()

    entropy = torch.mean(-torch.sum(probs * torch.log(probs + 1e-10), dim=1))

    return - torch.log(pos_sim / (pos_sim + neg_sim)) + entropy  # 0.1*loss_uni # 0.1

all_val = []
all_test = []
all_unf = []
all_f = []

loss_trains = []
loss_vals = []
for i in trange(args.runs, desc='Run Train'):
    # Model and optimizer
    model = FLAN_GCN(args=args,
                  nfeat=features.shape[1],
                  nhid=args.hidden,
                  nclass=labels.max().item() + 1,
                  dropout=args.dropout,
                  tau=args.tem)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.to(device)

    # Train model
    best = 999999999
    best_model = None
    best_X_list = None
    import time
    for epoch in range(args.epochs):
        s = time.time()
        model.train()
        optimizer.zero_grad()

        model_out, X1, X2 = model(features_normalized, adj_normalized, sparse_aligned_A)

        output = torch.log_softmax(model_out, dim=-1)

        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_con = conresive_loss(X1, X2, F_mask)

        loss_train = loss_train + args.lam*loss_con

        loss_train.backward()
        optimizer.step()

        model.eval()

        e = time.time()
        # print("time:", e-s)

        # val_X_list = get_augmented_features(args.concat)
        output, X1, X2 = model(features_normalized, adj_normalized, sparse_aligned_A)
        output = torch.log_softmax(output, dim=1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        test_acc = accuracy(output[idx_test], labels[idx_test])
        

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'test_acc: {:.4f}'.format(test_acc.item()),)

        if loss_val < best:
            best = loss_val
            best_model = copy.deepcopy(model)
            # best_X_list = copy.deepcopy(val_X_list)

        loss_vals.append(loss_val.item())
        loss_trains.append(loss_train.item())

    #Validate and Test
    best_model.eval()
    output, X1, X2 = best_model(features_normalized, adj_normalized, sparse_aligned_A)
    output = torch.log_softmax(output, dim=1)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    all_val.append(acc_val.item())
    all_test.append(acc_test.item())

    print(acc_test.item())

# print(np.mean(all_test),end=',')


print(np.mean(all_val), np.std(all_val), np.mean(all_test), np.std(all_test))

