# Unofficial implementation of 'All You Need Is Low (Rank): Defending Against Adversarial Attacks on Graphs' (https://dl.acm.org/doi/abs/10.1145/3336191.3371789)

import os
import torch
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import argparse
from ..model import *

from tqdm import tqdm
import pickle as pkl

from torch.nn import Parameter
from torch import Tensor
from torch_geometric.utils import add_self_loops, negative_sampling

import scipy.sparse as sp
def truncatedSVD(adj, k=50):
    """Truncated SVD on input adj.
    Parameters
    ----------
    adj :
        input matrix to be decomposed
    k : int
        number of singular values and vectors to compute.
    Returns
    -------
    numpy.array
        reconstructed matrix.
    """
    print('=== GCN-SVD: rank={} ==='.format(k))
    if sp.issparse(adj):
        adj = adj.asfptype()
        U, S, V = sp.linalg.svds(adj, k=k)
        print("rank_after = {}".format(len(S.nonzero()[0])))
        diag_S = np.diag(S)
    else:
        U, S, V = np.linalg.svd(adj)
        U = U[:, :k]
        S = S[:k]
        V = V[:k, :]
        print("rank_before = {}".format(len(S.nonzero()[0])))
        diag_S = np.diag(S)
        print("rank_after = {}".format(len(diag_S.nonzero()[0])))

    return U @ diag_S @ V

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--pretrain_model_dir', type=str, default='../saved_models')
    parser.add_argument('--dataset', type=str, default='synthetic')
    parser.add_argument('--dataset_argument', type=str, default='empty')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--flag', type=str, default='base')
    parser.add_argument('--increase', type=str, default='1')
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--node_mask', type=bool, default=False)
    parser.add_argument('--node_mask_ratio', type=float, default=.85)
    parser.add_argument('--base_lambda', type=float, default=1.)
    parser.add_argument('--start_lambda', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=0.98)
    args = parser.parse_args()
    
    if args.dataset == 'synthetic':
        with open(os.path.join(args.data_dir, args.dataset, '100_0.'+str(args.id)+'.pkl'), 'rb') as f:
            data = pkl.load(f)
    elif args.dataset in ['Photo', 'Computers']:
        dataset = Amazon(root=os.path.join(args.data_dir, 'Amazon'), name=args.dataset, transform=NormalizeFeatures())
        data = dataset[0]
        id_dir = os.path.join(args.data_dir, 'Amazon', args.dataset, 'ids')
    elif args.dataset == 'PolBlogs':
        dataset = PolBlogs(root=os.path.join(args.data_dir, args.dataset), transform=NormalizeFeatures())
        data = dataset[0]
        data.x = torch.ones((data.num_nodes,1), dtype=torch.float)
        id_dir = os.path.join(args.data_dir, args.dataset, 'ids')
    elif args.dataset in ['CS', 'Physics']:
        dataset = Coauthor(root=os.path.join(args.data_dir, 'Coauthor'), name=args.dataset, transform=NormalizeFeatures())
        data = dataset[0]
        id_dir = os.path.join(args.data_dir, 'Coauthor', args.dataset, 'ids')
    elif args.dataset in ['Cora','Citeseer','Pubmed']:
        dataset = Planetoid(root=os.path.join(args.data_dir, 'Planetoid'), split='full', name=args.dataset, transform=NormalizeFeatures())
        data = dataset[0]
        
    if args.dataset == 'synthetic':
        num_input_node_feature = 2
        num_output_class = 10
    elif args.dataset == 'PolBlogs':
        num_input_node_feature = 1
        num_output_class = dataset.num_classes
    else:
        num_input_node_feature = dataset.num_node_features
        num_output_class = dataset.num_classes
    
    if args.dataset in ['CS', 'Physics','PolBlogs', 'Photo', 'Computers']:
        train_ids = np.loadtxt(os.path.join(id_dir , 'train_'+str(args.id)+'.txt'), delimiter="\n", dtype='int')
        val_ids = np.loadtxt(os.path.join(id_dir , 'val_'+str(args.id)+'.txt'), delimiter="\n", dtype='int')
        test_ids = np.loadtxt(os.path.join(id_dir , 'test_'+str(args.id)+'.txt'), delimiter="\n", dtype='int')
        train, val, test = np.array([False]*data.num_nodes), np.array([False]*data.num_nodes), np.array([False]*data.num_nodes)
        test[test_ids] = True
        val[val_ids] = True
        train[train_ids] = True
        data.train_mask, data.val_mask, data.test_mask = torch.tensor(train), torch.tensor(val), torch.tensor(test)
    elif args.dataset in ['synthetic']:
        pass
    elif args.id != 1:
        raise Exception("For dataset " + args.dataset + ", id has to be 1.")
        
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    

    from torch_geometric.transforms import ToSparseTensor
    toSparse = ToSparseTensor(remove_edge_index=False)
    toSparse(data)
    svd_adj = truncatedSVD(sp.csr_matrix(data.adj_t.to_dense()), k=100)
    mask = np.abs(svd_adj) < 4e-2
    #normalized_svd_adj = (svd_adj - np.min(svd_adj))/np.ptp(svd_adj)
    svd_adj[mask] = 0
    from torch_geometric.utils import dense_to_sparse
    o, p = dense_to_sparse(torch.tensor(svd_adj))
    
    hidden_channels = args.hidden_channels
    lr = args.learning_rate
    num_layers = args.num_layers
    encoder = args.encoder
    seed = args.seed
    
    def train_gae(mask, edge_index, edge_weight):
        gae_model.train()
        optimizer_gae.zero_grad()  # Clear gradients.
        out = gae_model.encode_decode(data.x, edge_index, edge_weight)  # Perform a single forward pass.
        loss = criterion(out[mask], data.y[mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer_gae.step()  # Update parameters based on gradients.
        return loss

    def train_spcl(z, edge_index, gt_edge=None, _lambda=1., loss_type='increase', beta=1.):
        spcl_model.train()
        optimizer_spcl.zero_grad()  # Clear gradients.
        if gt_edge is None:
            gt_edge = torch.ones(len(edge_index[0]), device=z.device)
        # compute the reconstruction loss on original edges
        loss = spcl_model.recon_loss(z, edge_index, _lambda=_lambda, gt_edge=gt_edge, loss_type=loss_type, beta=beta) 
        loss.backward()  # Derive gradients.
        optimizer_spcl.step()  # Update parameters based on gradients.
        with torch.no_grad():
            for param in spcl_model.parameters():
                param.clamp_(0, 1)
        return loss

    def predict_gae(mask, edge_index, edge_weight):
        gae_model.eval()
        out = gae_model.encode_decode(data.x, edge_index, edge_weight)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
        return test_acc

    def predict_spcl(edge_index):
        spcl_model.eval()
        masked_edge_index, masked_edge_weight = spcl_model.structure_predict(edge_index) # compute the reconstruction loss on original edges
        return masked_edge_index, masked_edge_weight
    
    # initialize model
    if encoder == 'GCN':
        encoder_model = GCN(input_channels=num_input_node_feature, 
                            hidden_channels=hidden_channels, 
                            output_channels=hidden_channels,
                            num_layers=num_layers,
                            random_seed=seed
                           )
    elif encoder == 'GIN':
        encoder_model = GIN(input_channels=num_input_node_feature, 
                            hidden_channels=hidden_channels, 
                            output_channels=hidden_channels,
                            num_layers=num_layers,
                            random_seed=seed
                           )
    elif encoder == 'GraphSage':
        encoder_model = SAGE(input_channels=num_input_node_feature, 
                            hidden_channels=hidden_channels, 
                            output_channels=hidden_channels,
                            num_layers=num_layers,
                            random_seed=seed
                           )

    decoder_model = MLP(input_channels=hidden_channels, 
                        hidden_channels=hidden_channels, 
                        output_channels=num_output_class, 
                        random_seed=seed,
                        num_layers=1)
    gae_model = GAE(
        encoder=encoder_model, 
        decoder=decoder_model
    )

    # optimizer setting
    optimizer_gae = torch.optim.Adam(gae_model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # send to gpu
    gae_model = gae_model.to(device)
    data = data.to(device)
    o = o.to(device)
    p = p.to(device)

    if args.flag == 'svd':

        # base edge_index
        edge_index = o
        edge_weight = p

        # record lists
        train_acc_list = []
        val_acc_list = []
        test_acc_list = []
        loss_list = []
        num_edge_list = []
        num_epochs = args.epochs


        print('Epoch, Loss, Train Accuracy, Val Accuracy, Test Accuracy, Time')
        start_time = time.time()
        best_model = copy.deepcopy(gae_model)
        best_val = 0.
        for epoch in tqdm(range(1, num_epochs+1)):                
            # train main model
            loss = train_gae(data.train_mask, edge_index, edge_weight)
            loss_list.append(float(loss))

            # test model       
            train_acc = predict_gae(data.train_mask, edge_index, edge_weight)
            val_acc = predict_gae(data.val_mask, edge_index, edge_weight)
            test_acc = predict_gae(data.test_mask, edge_index, edge_weight)

            # record performance
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)

            # update best model
            if best_val < test_acc:
                best_model = copy.deepcopy(gae_model)
                best_val = test_acc

            print(f'{epoch:03d}, {loss:.4f}, {train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f} ,', (time.time() - start_time))


        # compute for best epoch based on validation performance
        val_acc_list = np.array(val_acc_list)
        test_acc_list = np.array(test_acc_list)
        max_index = np.argmax(val_acc_list)
        print(f'Best Epoch:{max_index:03d}, val acc:{val_acc_list[max_index]:.4f}, test acc:{test_acc_list[max_index]:.4f}')