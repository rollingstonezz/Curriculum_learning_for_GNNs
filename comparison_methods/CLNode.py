# Unofficial implementation of CLNode: Curriculum Learning for Node Classification (https://arxiv.org/abs/2206.07258)

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
    parser.add_argument('--alpha', type=float, default=1.)
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
        test_acc = int(test_correct.sum()) / (int(mask.sum())+1e-12)  # Derive ratio of correct predictions.
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
                        num_layers=1,
                        random_seed=seed)
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
    
    # trained model
    trained_encoder_model = GCN(input_channels=num_input_node_feature, 
                            hidden_channels=64, 
                            output_channels=64,
                            num_layers=num_layers
                           )

    trained_decoder_model = MLP(input_channels=64, 
                        hidden_channels=64, 
                        output_channels=num_output_class, 
                        num_layers=1)
    trained_gae_model = GAE(
        encoder=trained_encoder_model, 
        decoder=trained_decoder_model
    )
    save_model_dir = os.path.join(args.pretrain_model_dir, args.dataset)
    trained_gae_model.load_state_dict(torch.load(os.path.join(save_model_dir, 'GCN_'+str(args.id)+'.txt')))
    trained_gae_model.to(device)
    
    from torch_geometric.utils import add_remaining_self_loops, to_dense_adj
    with torch.no_grad():
        out = trained_gae_model.encode_decode(data.x, data.edge_index, torch.ones(data.num_edges, device=device))
        pred = out.argmax(dim=1)
        mixed_labels = copy.deepcopy(data.y)
        mixed_labels[~data.train_mask] = pred[~data.train_mask]

        # diversity scores
        one_hot_mixed_labels = torch.nn.functional.one_hot(mixed_labels)
        adj = to_dense_adj(data.edge_index)
        neighbor_labels = torch.matmul(adj[0], one_hot_mixed_labels.float())
        p = neighbor_labels / neighbor_labels.sum(axis=1).view(-1,1)
        diversity_score = - (p * torch.log(p+1e-12)).sum(axis=1)

        # consistency scores
        consistency_score = 1 - (neighbor_labels * one_hot_mixed_labels.float()).sum(axis=1) / neighbor_labels.sum(axis=1)

        # final scores 
        final_score = diversity_score + args.alpha * consistency_score
        final_score = final_score / final_score.max()
        
    # base edge_index
    edge_index = data.edge_index
    edge_weight = torch.ones(data.num_edges, device=device)

    # record lists
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    loss_list = []
    num_edge_list = []
    num_epochs = args.epochs

    print('Epoch, Loss, Train Accuracy, Val Accuracy, Test Accuracy, # of nodes, Time')
    start_time = time.time()
    best_model = copy.deepcopy(gae_model)
    best_val = 0.
    for epoch in tqdm(range(1, num_epochs+1)):         
        # calculate mask
        threshold = epoch / num_epochs
        mask = final_score <= threshold

        # train main model
        loss = train_gae(data.train_mask, edge_index, edge_weight)
        loss_list.append(float(loss))

        # test model       
        train_acc = predict_gae(data.train_mask & mask, edge_index, edge_weight)
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

        number_of_training_nodes = (data.train_mask & mask).sum()
        print(f'{epoch:03d}, {loss:.4f}, {train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f} ,{number_of_training_nodes:5d}', (time.time() - start_time))


    # compute for best epoch based on validation performance
    val_acc_list = np.array(val_acc_list)
    test_acc_list = np.array(test_acc_list)
    max_index = np.argmax(val_acc_list)
    print(f'Best Epoch:{max_index:03d}, val acc:{val_acc_list[max_index]:.4f}, test acc:{test_acc_list[max_index]:.4f}')