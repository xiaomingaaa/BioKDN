from dgl.nn.pytorch.conv import GraphConv as GCNConv, GATConv, GINConv
from dgl.nn.pytorch.glob import AvgPooling, SumPooling
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli, LogitRelaxedBernoulli
from subgraph_util.graph_utils import normalize_adj

VERY_SMALL_NUMBER = 1e-12
INF = 1e20

class GraphLearner(nn.Module):
    def __init__(self, params) -> None:
        super(GraphLearner, self).__init__()
        self.input_size = params.input_size
        self.hidden_size = params.embed_dim
        self.num_pers = params.num_pers
        self.metric_type = params.metric_type
        self.graph_type = params.graph_type
        self.feature_denoise = params.feature_denoise
        # self.device = device
        self.top_k = params.top_k

        if params.metric_type == 'attention':
            self.linear_sims = nn.ModuleList([nn.Linear(self.input_size, params.hidden_size, bias=False) for _ in range(params.num_pers)])
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(params.metric_type, params.num_pers))
        elif params.metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(params.num_pers, self.input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(params.metric_type, params.num_pers))
        elif params.metric_type == 'gat_attention':
            self.linear_sims1 = nn.ModuleList([nn.Linear(self.input_size, 1, bias=False) for _ in range(params.num_pers)])
            self.linear_sims2 = nn.ModuleList([nn.Linear(self.input_size, 1, bias=False) for _ in range(params.num_pers)])
            self.leakyrelu = nn.LeakyReLU(0.2)
            print('[ GAT_Attention GraphLearner]')
        elif params.metric_type == 'kernel':
            self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.input_size, self.hidden_size)))
        elif params.metric_type == 'transformer':
            self.linear_sim1 = nn.Linear(self.input_size, self.hidden_size, bias=False)
            self.linear_sim2 = nn.Linear(self.input_size, self.hidden_size, bias=False)
        elif params.metric_type == 'cosine':
            pass
        elif params.metric_type == 'mlp':
            self.lin1 = nn.Linear(self.input_size, self.hidden_size)
            self.lin2 = nn.Linear(self.hidden_size, self.hidden_size)
        elif params.metric_type == 'multi_mlp':
            self.linear_sims1 = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size, bias=False) for _ in range(params.num_pers)])
            self.linear_sims2 = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(params.num_pers)])
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(params.metric_type, params.num_pers))
        if self.feature_denoise:
            self.feat_mask = self.construct_feat_mask(self.input_size, init_strategy="constant")

    def reset_parameter(self):
        if self.metric_type == 'attention':
            for module in self.linear_sims:
                module.reset_parameters()
        
    def forward(self, node_features):
        if self.feature_denoise:
            masked_features = self.mask_feature(node_features)
            learned_adj = self.learn_adj(masked_features)
            return masked_features, learned_adj
        else:
            learned_adj = self.learn_adj(node_features)
            return node_features, learned_adj
    
    def learn_adj(self, context, ctx_mask=None):
        if self.metric_type == 'attention':
            attention = 0
            for _ in range(len(self.linear_sims)):
                context_fc = torch.relu(self.linear_sims[_](context))
                attention += torch.matmul(context_fc, context_fc.transpose(-1, -2))

            attention /= len(self.linear_sims)
            markoff_value = -INF
        elif self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            markoff_value = 0

        elif self.metric_type == 'transformer':
            Q = self.linear_sim1(context)
            attention = torch.matmul(Q, Q.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
            markoff_value = -INF

        elif self.metric_type == 'gat_attention':
            attention = []
            for _ in range(len(self.linear_sims1)):
                a_input1 = self.linear_sims1[_](context)
                a_input2 = self.linear_sims2[_](context)
                attention.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))

            attention = torch.mean(torch.stack(attention, 0), 0)
            markoff_value = -INF
            # markoff_value = 0

        elif self.metric_type == 'kernel':
            dist_weight = torch.mm(self.weight, self.weight.transpose(-1, -2))
            attention = self.compute_distance_mat(context, dist_weight)
            attention = torch.exp(-0.5 * attention * (self.precision_inv_dis**2))

            markoff_value = 0

        elif self.metric_type == 'cosine':
            context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
            attention = torch.mm(context_norm, context_norm.transpose(-1, -2)).detach()
            markoff_value = 0
        elif self.metric_type == 'mlp':
            context_fc = torch.relu(self.lin2(torch.relu(self.lin1(context))))
            attention = torch.matmul(context_fc, context_fc.transpose(-1, -2))
            markoff_value = 0
        elif self.metric_type == 'multi_mlp':
            attention = 0
            for _ in range(self.num_pers):
                context_fc = torch.relu(self.linear_sims2[_](torch.relu(self.linear_sims1[_](context))))
                attention += torch.matmul(context_fc, context_fc.transpose(-1, -2))

            attention /= self.num_pers
            markoff_value = -INF
        
        if self.graph_type == 'epsilonNN':
            assert self.epsilon is not None
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)
        elif self.graph_type == 'KNN':
            assert self.top_k is not None
            attention = self.build_knn_neighbourhood(attention, self.top_k, markoff_value)
        elif self.graph_type == 'prob':
            attention = self.build_prob_neighbourhood(attention, temperature=0.05)
        else:
            raise ValueError('Unknown graph_type: {}'.format(self.graph_type))
        if self.graph_type in ['KNN', 'epsilonNN']:
            if self.metric_type in ('kernel', 'weighted_cosine'):
                assert attention.min().item() >= 0
                attention = attention / torch.clamp(torch.sum(attention, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
            elif self.metric_type == 'cosine':
                attention = (attention > 0).float()
                attention = normalize_adj(attention)
            elif self.metric_type in ('transformer', 'attention', 'gat_attention'):
                attention = torch.softmax(attention, dim=-1)

        return attention

    def build_knn_neighbourhood(self, attention, top_k, markoff_value):
        top_k = min(top_k, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, top_k, dim=-1)
        weighted_adjacency_matrix = (markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val)
        weighted_adjacency_matrix = weighted_adjacency_matrix

        return weighted_adjacency_matrix

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        attention = torch.sigmoid(attention)
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def build_prob_neighbourhood(self, attention, temperature=0.1):
        attention = torch.clamp(attention, 0.01, 0.99)

        weighted_adjacency_matrix = RelaxedBernoulli(temperature=torch.Tensor([temperature]).to(attention.device),
                                                     probs=attention).rsample()
        eps = 0.5
        mask = (weighted_adjacency_matrix > eps).detach().float()
        weighted_adjacency_matrix = weighted_adjacency_matrix * mask + 0.0 * (1 - mask)
        return weighted_adjacency_matrix

    def compute_distance_mat(self, X, weight=None):
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X
        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)
        return dists

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def mask_feature(self, x, use_sigmoid=True, marginalize=True):
        feat_mask = (torch.sigmoid(self.feat_mask) if use_sigmoid else self.feat_mask)
        if marginalize:
            std_tensor = torch.ones_like(x, dtype=torch.float) / 2
            mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
            z = torch.normal(mean=mean_tensor, std=std_tensor)
            x = x + z * (1 - feat_mask)
        else:
            x = x * feat_mask
        return x

class GCN(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, embed_dim)
        self.conv2 = GCNConv(embed_dim, embed_dim)
        self.pool = AvgPooling()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, g, h):
        
        x = F.relu(self.conv1(g, h))
        skip_conn1 = self.pool(g, x)
        x = F.relu(self.conv2(g, x))
        skip_conn2 = self.pool(g, x)
        x = self.pool(g, x)
        output = skip_conn1 + skip_conn2 + x
        
        return output

    def __repr__(self):
        return self.__class__.__name__

class GIN(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(GIN, self).__init__()
        self.lin1 = nn.Linear(input_dim, embed_dim)
        self.conv1 = GINConv(self.lin1, 'sum', activation=F.relu)
        self.conv2 = GINConv(self.lin1, 'sum')
        self.pool = AvgPooling()

    def forward(self, g, h, edge_wight):
        x = F.relu(self.conv1(g, h, edge_wight))
        x = F.relu(self.conv2(g, x, edge_wight))
        x = self.pool(g, x)

        output = x
        return output

    def __repr__(self):
        return self.__class__.__name__


class myGCN(nn.Module):
    def __init__(self, args):
        super(myGCN, self).__init__()
        self.conv1 = GCNConv(args.input_size, args.embed_dim, allow_zero_in_degree=True)
        self.conv2 = GCNConv(args.embed_dim, args.embed_dim, allow_zero_in_degree=True)
        self.lin1 = nn.Linear(args.input_size, args.embed_dim)
        self.pool = AvgPooling()

    def forward(self, g, h):
        x = self.conv1(g, h)
        # out = self.pool(g, x)
        x = F.relu(x)
        x = self.conv2(g, x)
        h = self.lin1(h)
        x = self.pool(g, x + h)
        
        return x

    def __repr__(self):
        return self.__class__.__name__


class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)
 
    def to(self,device):
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")
