import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.layers import GraphLearner, GCN, myGCN, GIN
import dgl
from scipy.sparse import csr_matrix

class GraphClassifier(nn.Module):
    def __init__(self, args, drug_dims, protein_dims, dropout=0.5):
        super(GraphClassifier,self).__init__()
        self.drug_dims=drug_dims
        self.protein_dims=protein_dims
        self.dropout=dropout
        self.args = args
        if args.split_type == 'multi_class':
            self.types = 87
            self.logit_func = F.softmax
            self.loss_func = F.cross_entropy
        else:
            self.types = 1
            self.logit_func = F.sigmoid
            self.loss_func = F.binary_cross_entropy
            
        if args.kge:
            entity_embed, relation_embed = args.load_kge()
            self.entity_embed = nn.Parameter(torch.from_numpy(entity_embed), requires_grad=False)
            self.relation_embed = nn.Parameter(torch.from_numpy(relation_embed), requires_grad=False)
        else:
            self.entity_embed = nn.Embedding(93787, args.embed_dim)
            self.relation_embed = nn.Embedding(107, args.embed_dim)
        self.graph_learner = GraphLearner(args)
        self.gnn = GCN(args.input_size, args.embed_dim)
        self.backbone = myGCN(args)
        self.drug_net=nn.ModuleList()
        self.protein_net=nn.ModuleList()
        self.cls = nn.ModuleList()
        self._construct_DNN()
        print('types:', self.types)
        
        

    def _construct_DNN(self):
        for i in range(1, len(self.drug_dims)):
            self.drug_net.append(nn.Linear(self.drug_dims[i-1],self.drug_dims[i]))
            self.drug_net.append(nn.BatchNorm1d(self.drug_dims[i]))
            self.drug_net.append(nn.Dropout(self.dropout))
            self.drug_net.append(nn.ReLU())
            
        # self.drug_net.pop(-1)
        for i in range(1, len(self.protein_dims)):
            self.protein_net.append(nn.Linear(self.protein_dims[i-1],self.protein_dims[i]))
            self.protein_net.append(nn.BatchNorm1d(self.protein_dims[i]))
            self.protein_net.append(nn.Dropout(self.dropout))
            self.protein_net.append(nn.ReLU())
            
        # self.protein_net.pop(-1)

        self.cls.append(nn.Linear(0*self.drug_dims[-1]+0*self.protein_dims[-1]+2*self.args.embed_dim, 100))
        self.cls.append(nn.ReLU())
        self.cls.append(nn.Linear(100, self.types))

    def batch_graph_learner(self, batch_g):
        graphs = []
        old_graphs = []
        for graph in batch_g:
            
            node_features, learned_adj = self.graph_learner(self.entity_embed[graph.ndata[dgl.NID]])
            csr_adj = csr_matrix(learned_adj.detach().cpu().numpy())
            learned_g = dgl.add_self_loop(dgl.from_scipy(csr_adj, eweight_name='e_w'))
            learned_g.ndata['h'] = node_features.detach().cpu()
            graphs.append(learned_g)
            graph.ndata['h'] = node_features.detach().cpu()
            old_graphs.append(graph)
            
        return dgl.batch(graphs).to(self.args.device), dgl.batch(old_graphs).to(self.args.device)
    
    def contrastive_Loss(self, src, dst):
        scores = torch.matmul(self.norm(src), self.norm(dst.T))
        labels = torch.eye(scores.shape[0]).to(self.args.device)

        logits = F.sigmoid(scores)
        c_loss = F.binary_cross_entropy(logits.view(1, -1), labels.view(1, -1))

        return c_loss

    def norm(self, x):
        x = F.normalize(x, p=2)
        return x

    def reparametrize_n(self, mu, std, n=1):
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps * std

    def forward(self, protein_embed, drug_embed, labels, graphs, h_gs):
        g, old_g = self.batch_graph_learner(graphs)
        g_out = self.gnn(g, g.ndata['h'])
        old_g_backbone = self.backbone(old_g, old_g.ndata['h'])
        g = dgl.batch(graphs).to(self.args.device)
        h_g_out = self.gnn(h_gs, self.entity_embed[h_gs.ndata[dgl.NID]])
        c_loss = self.contrastive_Loss(g_out, h_g_out)
        
        if self.types==1:
            labels = labels.float().unsqueeze(1)
        else:
            labels = labels.long()
        for l in self.protein_net:
            protein_embed = l(protein_embed)
        
        for l in self.drug_net:
            drug_embed = l(drug_embed)
        
        embed = torch.concat([protein_embed, drug_embed, self.norm(old_g_backbone), self.norm(g_out), self.norm(h_g_out)], dim=1)
        
        for layer in self.cls:
            embed=layer(embed)
        class_loss=self.loss_func(embed, labels)
        loss = class_loss + c_loss
        return loss, self.logit_func(embed, dim=-1), labels

