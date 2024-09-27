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
import time

from utils import load_data, accuracy, get_dataset, row_l1_normalize, GCN_norm, get_dataset_split, pairwise_distance
from gcn.models2 import LAGCN, CLAGCN, GCN, LP
from gat.models import GAT
from tqdm import trange
from visualize import visualize_weights

exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument("--concat", type=int, default=4)
parser.add_argument('--runs', type=int, default=10, help='The number of experiments.')

parser.add_argument('--dataset', default='pubmed', help='Dataset string.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--edge_rate', type=float, default=0.5, help='sparse feature distance matrix.')

parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1.2, help='Lamda') # sample=1时，consis_loss相当于就sharp了一下
parser.add_argument('--I', type=bool, default=False, help='Lamda')
parser.add_argument('--k', type=int, default=20, help=' K for knn.')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

if args.I:
    args.I = 1.0
else:
    args.I = 0.0


path = "data"
# Load data
# adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)
dataset, evaluator = get_dataset(
    name=args.dataset,
    root_dir=path,
)
data = dataset._data.to(device)
adj, features, idx_train, idx_val, idx_test, labels = data.edge_index, data.x, data.train_mask, data.val_mask, data.test_mask, data.y
features_normalized = row_l1_normalize(features)
adj_normalized = GCN_norm(adj, features)

# knn_adj_norm = torch.load(f'./knn_graph/{args.dataset}_sims_{args.k}.pt')


if args.cuda:
    adj_normalized = adj_normalized.to(device)
    labels = labels.squeeze().long().to(device)
    features_normalized = features_normalized.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

# feature distance
feature_distance = pairwise_distance(features)
feature_distance = F.normalize(feature_distance)
kthvalue = torch.kthvalue(
    feature_distance.view(feature_distance.shape[0] * feature_distance.shape[1], 1).T,
    int(feature_distance.shape[0] * feature_distance.shape[1] * args.edge_rate))[0]
mask = (feature_distance > kthvalue).detach().float()
feature_distance = (feature_distance * mask)
# print(feature_distance)
# exit()


all_val = []
all_test = []
all_unf = []
all_f = []

all_start = time.time()
loss_trains = []
loss_vals = []
for i in trange(args.runs, desc='Run Train'):
    # Model and optimizer
    model = GCN(args=args,
                  nfeat=features.shape[1],
                  nhid=args.hidden,
                  nclass=labels.max().item() + 1,
                  dropout=args.dropout,
                  tau=args.tem)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # model = GAT(
    #               nfeat=features.shape[1],
    #               nhid=args.hidden,
    #               nclass=labels.max().item() + 1,
    #               dropout=args.dropout)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

        model_out = model(features_normalized, adj_normalized)

        output = torch.log_softmax(model_out, dim=-1)

        loss_train = F.nll_loss(output[idx_train], labels[idx_train])

        loss_train = loss_train

        loss_train.backward()
        optimizer.step()

        model.eval()

        e = time.time()
        # print("time:", e-s)

        # val_X_list = get_augmented_features(args.concat)
        output = model(features_normalized, adj_normalized)
        output = torch.log_softmax(output, dim=1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        test_acc = accuracy(output[idx_test], labels[idx_test])
        

        # print('Epoch: {:04d}'.format(epoch+1),
        #       'loss_train: {:.4f}'.format(loss_train.item()),
        #       'loss_val: {:.4f}'.format(loss_val.item()),
        #       'test_acc: {:.4f}'.format(test_acc.item()),)

        if loss_val < best:
            best = loss_val
            best_model = copy.deepcopy(model)
            # best_X_list = copy.deepcopy(val_X_list)

        loss_vals.append(loss_val.item())
        loss_trains.append(loss_train.item())

    #Validate and Test
    best_model.eval()
    output = best_model(features_normalized, adj_normalized)
    output = torch.log_softmax(output, dim=1)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    all_val.append(acc_val.item())
    all_test.append(acc_test.item())

    print(acc_test.item())

    all_end = time.time()
    # print("总时间:", all_end - all_start)




    F_mask = torch.load("./UNF_mask/" + args.dataset + "_F_mask.pt")
    acc_UNF = accuracy(output[idx_test & (~F_mask)], labels[idx_test & (~F_mask)])
    acc_F = accuracy(output[idx_test & (F_mask)], labels[idx_test & (F_mask)])

    torch.save(F_mask, "./UNF_mask/"+args.dataset+"_F_mask.pt")

    torch.save(output.max(1)[1], "./aligned_nodes/" + args.dataset + "_aligned_label.pt")

    exit()



print(np.mean(all_test),end=',')


print(np.mean(all_val), np.std(all_val), np.mean(all_test), np.std(all_test))

