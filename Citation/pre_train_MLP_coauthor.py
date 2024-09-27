import numpy as np

import torch
import torch.nn.functional as F
import random

from Citation.utils import accuracy, get_dataset, get_dataset_split, row_l1_normalize
from Citation.gcn.models2 import MLP, LP
from Citation.early_stop import EarlyStopping, Stop_args
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='squirrel',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--num_epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, # 0.005
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--patience', type=int, default=400,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--num_layers', type=int, default=2)

args = parser.parse_args()
# torch.manual_seed(0)

args.seed = 0
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def train_MLP(args, data, train_mask, val_mask, test_mask):
    # Model and optimizer
    # print(MLP)
    # model = MLP(args.num_features, args.hidden_dim, args.num_classes, args.num_layers, args.dropout).to(device)
    model = MLP(args.num_features, args.hidden_dim, args.num_classes, args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)

    def train(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(data.x)
        # print(output.shape)
        # print(data.y.shape)
        loss_train = F.nll_loss(output[train_mask],
                                data.y[train_mask])  # + args.weight_decay * torch.sum(model.linear1.weight ** 2) / 2
        loss_train.backward()
        optimizer.step()

        acc_train = accuracy(output[train_mask], data.y[train_mask])

        # Evaluate validation set performance separately,
        model.eval()
        output = model(data.x)

        loss_val = F.nll_loss(output[val_mask], data.y[val_mask])
        acc_val = accuracy(output[val_mask], data.y[val_mask])

        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.4f}'.format(loss_train.item()),
        #       'acc_train: {:.4f}'.format(acc_train.item()),
        #       'loss_val: {:.4f}'.format(loss_val.item()),
        #       'acc_val: {:.4f}'.format(acc_val.item()))

        return loss_val.item(), acc_val.item()

    def A_test():
        model.eval()
        output = model(data.x)
        loss_test = F.nll_loss(output[test_mask], data.y[test_mask])
        acc_test = accuracy(output[test_mask], data.y[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item(), output.exp()

    stopping_args = Stop_args(patience=args.patience, max_epochs=args.num_epochs)
    early_stopping = EarlyStopping(model, **stopping_args)
    for epoch in range(args.num_epochs):
        loss_val, acc_val = train(epoch)
        if early_stopping.check([acc_val, loss_val], epoch):
            break

    print("Optimization Finished!")

    # Restore best model
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    model.load_state_dict(early_stopping.best_state)
    acc_test, logits_MLP = A_test()
    print(acc_test)
    return model, logits_MLP


def pick_unfriendly_nodes(MLP_model, train_mask, data):

    MLP_model.eval()
    # print(MLP_model.training)
    x, y, edge_index = data.x, data.y.long(), data.edge_index
    row, col = edge_index
    num_nodes = x.shape[0]
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    adj_norm = gcn_norm(adj.to_symmetric(), add_self_loops=True)
    # adj_norm = MLP_model.adj_norm
    logits_MLP = MLP_model(x).exp()  # since utilize the log softmax

    if args.dataset in ["citeseer", "pubmed", "photo"]:
        lp = LP(y, train_mask, K=5, alpha=0.9)  # for citeseer,K=5, cora=3
    else:
        lp = LP(y, train_mask, K=3, alpha=0.9)  # for citeseer,K=5, cora=3
    logits_lp = lp(adj_norm)


    # #
    # UNF_mask = torch.load("./UNF_mask/" + args.dataset + "_UNF_mask.pt")
    # acc_UNF = accuracy(logits_lp[UNF_mask], y[UNF_mask])
    # print("UN_F acc:", acc_UNF.item())
    # exit()


    # For point 1
    MASK1 = logits_lp.max(1)[1] == (adj_norm@adj_norm@logits_MLP).max(1)[1].to(device) # logits_MLP.max(1)[1].to(device)

    return MASK1, adj_norm@adj_norm@logits_MLP, logits_lp

def confidence_gap(logits):
    temp_max = torch.max(logits, 1)[0].reshape(-1, 1)
    b = (temp_max - logits)
    # print(b)
    # exit()
    c_gap = torch.min(torch.where(b == 0, 1 , b), 1)[0]
    return c_gap


def index2bool(index, len):
    if max(index) > len:
        print("error! index out of mask!")
        exit()
    mask = torch.zeros(len).to(device)
    mask[index] = 1
    return mask.bool()


path = "data"
# Load data
# adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)
dataset, evaluator = get_dataset(
    name=args.dataset,
    root_dir=path,
)
data = dataset._data.to(device)

epoch = 0
for i in range(10):


    adj, features,labels = data.edge_index, data.x,  data.y.long().squeeze()
    idx_train, idx_val, idx_test = get_dataset_split(args.dataset, data, None, i)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    data.y = data.y.long().squeeze()
    # data.x = row_l1_normalize(data.x)
    args.num_features = features.shape[1]
    args.num_classes = max(labels)+1
    mlp, logits_MLP = train_MLP(args, data, idx_train, idx_val, idx_test)

    torch.save(logits_MLP, "./X_MLP/" + args.dataset + "_X_MLP"+str(i)+".pt")
    # print(logits_MLP)
    # exit()


    F_mask, logits_SGC, logits_lp = pick_unfriendly_nodes(mlp, idx_train, data)
    idx_test = idx_test.to(device)
    UNF_mask = ~F_mask

    acc_SGC = (data.y == logits_SGC.max(1)[1])[UNF_mask&(~idx_train)].int().sum()/((UNF_mask&(~idx_train)).int().sum())
    print(acc_SGC)

    acc_MLP = (data.y == logits_SGC.max(1)[1])[idx_test].int().sum()/(idx_test.int().sum())
    print(acc_MLP)

    pseudo_labels = logits_SGC.max(1)[1]

    # k_lable = torch.zeros_like(aligned_nodes)
    # k_lable[F_mask] = aligned_nodes[F_mask]
    torch.save(pseudo_labels, "./aligned_nodes/"+args.dataset+"_aligned_label"+str(i)+".pt")


    if args.dataset in ["cora", "citeseer", "pubmed"]:
        torch.save(UNF_mask, "./UNF_mask/"+args.dataset+"_UNF_mask.pt")
        torch.save(F_mask, "./UNF_mask/"+args.dataset+"_F_mask.pt")
    else:
        torch.save(UNF_mask, "./UNF_mask/" + args.dataset + "_UNF_mask"+str(i)+".pt")
        torch.save(F_mask, "./UNF_mask/" + args.dataset + "_F_mask"+str(i)+".pt")