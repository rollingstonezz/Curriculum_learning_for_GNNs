##########################################################################################
#############The code is modified from pytorch geometric library##########################
########https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html################
##########################################################################################

import torch
from layers import SAGEConv, GINConv, GCNConv
from torch_geometric.nn import BatchNorm
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU
from torch_geometric.nn.inits import reset
from torch_geometric.utils import add_self_loops, negative_sampling

class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, random_seed=1234567, num_layers=2):
        super().__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)
        self.convs = ModuleList()
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.num_layers = num_layers
        
    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        if self.num_layers == 1:
            return x
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
class GIN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, random_seed=1234567, num_layers=2):
        super().__init__()
        torch.manual_seed(random_seed)
        nn1 = MLP(input_channels, hidden_channels, hidden_channels, num_layers=2, random_seed=random_seed)
        nn2 = MLP(hidden_channels, hidden_channels, output_channels, num_layers=2, random_seed=random_seed)
        self.conv1 = GINConv(nn=nn1, eps=0.1)
        self.conv2 = GINConv(nn=nn2, eps=0.1)
        self.convs = ModuleList()
        for _ in range(num_layers-2):
            nn_cur = MLP(hidden_channels, hidden_channels, hidden_channels, num_layers=2, random_seed=random_seed)
            self.convs.append(GINConv(nn=nn_cur, eps=0.1))
        self.num_layers = num_layers

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        if self.num_layers == 1:
            return x
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
class SAGE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, random_seed=1234567, num_layers=2):
        super().__init__()
        torch.manual_seed(random_seed)
        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, output_channels)
        self.convs = ModuleList()
        for _ in range(num_layers-2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.num_layers = num_layers

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        if self.num_layers == 1:
            return x
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, random_seed=1234567, num_layers=2):
        super().__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GATConv(input_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, output_channels)
        self.convs = ModuleList()
        for _ in range(num_layers-2):
            self.convs.append(GATConv(hidden_channels, hidden_channels))
        self.num_layers = num_layers

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        if self.num_layers == 1:
            return x
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
class MLP(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers=2, random_seed=1234567):
        super(MLP, self).__init__()
        torch.manual_seed(random_seed)
        self.lin_list = ModuleList()
        if num_layers == 1:
            self.lin_list.append(Linear(input_channels, output_channels, bias=False))
        else:
            self.lin_list.append(Linear(input_channels, hidden_channels, bias=False))
            for _ in range(num_layers-2):
                self.lin_list.append(Linear(hidden_channels, hidden_channels, bias=False))
            self.lin_list.append(Linear(hidden_channels, output_channels, bias=False))

    def forward(self, x):
        x = self.lin_list[0](x)
        for lin in self.lin_list[1:]:
            x = x.relu()
            x = lin(x)
        return x


class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

class CosineDecoder(torch.nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        z_norm = z / z.norm(dim=1)[:, None]
        cosine_res = (z_norm[edge_index[0]] * z_norm[edge_index[1]]).sum(dim=1)
        return (cosine_res+1)/2 if sigmoid else cosine_res

class GAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def encode_decode(self, *args, **kwargs):
        z = self.encoder(*args, **kwargs)
        return self.decoder(z)

class SPCL(torch.nn.Module):
    def __init__(self, num_edges, structure_decoder=None):
        super().__init__()
        self.s_mask = Parameter(torch.zeros(num_edges, dtype=torch.float))
        self.accumulated_s_mask = torch.zeros(num_edges, dtype=torch.float, device="cuda:0")
        self.num = 1e-8
        self.structure_decoder = InnerProductDecoder() if structure_decoder is None else structure_decoder
        SPCL.reset_parameters(self)

    def reset_parameters(self):
        reset(self.s_mask)
        reset(self.structure_decoder)

    def recon_loss(self, z, edge_index, _lambda, gt_edge, loss_type='increase', beta=1.):

        # predicted structure 
        pred_struct = self.structure_decoder(z, edge_index).view(-1)
        
        # compute loss 
        if loss_type == 'increase':
            structure_loss = torch.sum(
                self.s_mask*(pred_struct-gt_edge)**2
            ) - _lambda*torch.sum(self.s_mask)
        elif loss_type == 'both':
            term1 = -torch.mean(
                self.s_mask*pred_struct
            )
            term2 = beta*torch.mean(self.s_mask)
            term3 = _lambda*torch.mean((self.s_mask - gt_edge)**2)
            structure_loss = term1 + term2 + term3
        return structure_loss

    def structure_predict(self, edge_index):
        with torch.no_grad():
            mask = self.s_mask > 0.5
            self.accumulated_s_mask = mask.detach().float() + self.accumulated_s_mask
            masked_edge_index = edge_index.T[mask].T
            self.num += 1
            masked_edge_weight = self.accumulated_s_mask[mask] / self.num
            return masked_edge_index, masked_edge_weight