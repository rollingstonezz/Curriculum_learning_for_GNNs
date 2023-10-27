import os
import torch
from torch_geometric.datasets import MixHopSyntheticDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle as pkl
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()
    
    dim = 2
    N = 500
    num_classes = 10
    R = 300
    r = 100
    E = 30000
    for homo_ratio in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        sampled_points = []
        for t in range(num_classes):
            center_point = [R*np.sin(np.pi/180*t/num_classes*360), R*np.cos(np.pi/180*t/num_classes*360)]
            sampled_points.append(
                np.random.multivariate_normal(
                    mean=center_point, 
                    cov=[[r**2,0],[0,r**2]], 
                    size=N
                )
            )
        # build edge sample prob matrix
        dist_2d = [[0]*num_classes for _ in range(num_classes)]
        prob_2d = [[0]*num_classes for _ in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                temp1 = abs(np.arange(num_classes)[i] - np.arange(num_classes)[j])
                temp2 = abs(num_classes + np.arange(num_classes)[i] - np.arange(num_classes)[j])
                temp3 = abs(num_classes + np.arange(num_classes)[j] - np.arange(num_classes)[i])
                dist_2d[i][j] = min(temp1, temp2, temp3)
                prob_2d[i][j] = 1. / (dist_2d[i][j]+1e-12)
            prob_2d[i][i] = 0
            dist_2d[i][i] = 999

        prob_2d = prob_2d / np.sum(prob_2d)

        unsample_pairs = []
        for i in range(num_classes):
            for j in range(num_classes):
                unsample_pairs.append((i,j))

        edges_dict = {}
        edge_prob_list = []
        for _ in tqdm(range(E)):
            if np.random.random() < homo_ratio:
                # homo class
                chosen_class = np.random.choice(np.arange(num_classes))
                sample_pair = np.random.choice(np.arange(N), 2, replace=False) + chosen_class*N
                while tuple(sample_pair) in edges_dict:
                    sample_pair = np.random.choice(np.arange(N), 2, replace=False) + chosen_class*N
                edges_dict[tuple(sample_pair)] = True
                edges_dict[tuple(sample_pair[::-1])] = True
                edge_prob_list.append(1.)
                edge_prob_list.append(1.)
            else:
                # non-homo class
                cluster_1, cluster_2 = unsample_pairs[np.random.choice(
                    np.arange(len(unsample_pairs)), 
                    p=prob_2d.reshape(-1), 
                    replace=False)]

                sample_pair = np.random.choice(np.arange(N)) + cluster_1*N, np.random.choice(np.arange(N)) + cluster_2*N
                while tuple(sample_pair) in edges_dict:
                    sample_pair = np.random.choice(np.arange(N), 2, replace=False) + chosen_class*N
                edges_dict[tuple(sample_pair)] = True
                edges_dict[tuple(sample_pair[::-1])] = True
                edge_prob_list.append(prob_2d[cluster_1][cluster_2])
                edge_prob_list.append(prob_2d[cluster_2][cluster_1])

        x = np.concatenate(sampled_points)
        edge_index = np.array(list(edges_dict.keys())).T
        y = np.arange(num_classes*N) // N

        train_mask, val_mask, test_mask = [False] * (num_classes*N), [False] * (num_classes*N), [False] * (num_classes*N)
        random_shuffle = np.random.choice(num_classes*N, num_classes*N, replace=False)

        for item in random_shuffle[:num_classes*N//3+1]:
            train_mask[item] = True
        for item in random_shuffle[num_classes*N//3+1:num_classes*N*2//3]:
            val_mask[item] = True
        for item in random_shuffle[num_classes*N*2//3:]:
            test_mask[item] = True

        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor(y, dtype=torch.long),
            train_mask=torch.tensor(train_mask, dtype=torch.bool),
            val_mask=torch.tensor(val_mask, dtype=torch.bool),
            test_mask=torch.tensor(test_mask, dtype=torch.bool),
            edge_prob=torch.tensor(edge_prob_list, dtype=torch.float)
        )

        save_dir = os.path.join(args.data_dir, 'synthetic')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, str(r)+'_'+str(homo_ratio)+'.pkl'), 'wb') as f:
            pkl.dump(data, f)

        print(homo_ratio)
        print(data)

        # Gather some statistics about the graph.
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Number of training nodes: {data.train_mask.sum()}')
        print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
        print('===========================================================================================================')