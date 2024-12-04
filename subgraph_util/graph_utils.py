import pandas as pd
import dgl
import torch
import pickle
import numpy as np

def normalize_adj(mx):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

def collate_dgl(samples):
    protein, drug, label, subgraphs, h_sg = map(list, zip(*samples))
    h_sg = dgl.batch(h_sg)

    return (torch.tensor(drug), torch.tensor(protein), subgraphs, h_sg), torch.tensor(label) #, drug_idx

def load_kge_embed(model_name='TransE'):
    entity_embed = np.load('ckpts/pretrained_embed/DRKG_{}_entity.npy'.format(model_name))
    relation_embed = np.load('ckpts/pretrained_embed/DRKG_{}_relation.npy'.format(model_name))

    return entity_embed, relation_embed

def save_pickle(data, file_path):
    print('save data....')
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return  data

def process_files(args, files, triple_file, dataset='drugbank', max_num_nodes=300, root_path='/', hop=1):
    entity_dict = {}
    relation_dict = {}
    id2entity = {}
    with open('dataset/drkg/entity2id.tsv', 'r') as f:
        for line in f:
            entity, eid = line.strip().split('\t')
            entity_dict[entity] = int(eid)
            id2entity[int(eid)] = entity
    with open('dataset/drkg/relation2id.tsv', 'r') as f:
        for line in f:
            relation, rid = line.strip().split('\t')
            relation_dict[relation] = int(rid)

    src = []
    dst = []
    # edge_type = []
    edge_dict = {}
    node_type = ['Gene', 'Compound', 'Disease']
    entity_label = {'Compound':{}, 'Gene':{}, 'Disease':{}}
    hetro_entity_feat = {'Compound':[], 'Gene':[], 'Disease':[]}
    with open(triple_file, 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            h_l, t_l = h.split('::')[0], t.split('::')[0]
            if h_l in node_type and t_l in node_type:
                ## process hetero
                edge_type = (h_l, h_l+'_'+t_l, t_l)
                if h not in entity_label[h_l]:
                    entity_label[h_l][h] = len(entity_label[h_l])
                    hetro_entity_feat[h_l] += [entity_dict[h]]

                if t not in entity_label[t_l]:
                    entity_label[t_l][t] = len(entity_label[t_l])
                    hetro_entity_feat[t_l] += [entity_dict[t]]

                if edge_type not in edge_dict:
                    edge_dict[edge_type] = list()
                    edge_dict[edge_type].append((entity_label[h_l][h], entity_label[t_l][t]))
                else:
                    edge_dict[edge_type].append((entity_label[h_l][h], entity_label[t_l][t]))

            src.append(entity_dict[h])
            dst.append(entity_dict[t])
            # edge_type.append(relation_dict[r])
    
    hetro = dgl.heterograph(edge_dict)

    
    kg = dgl.graph((src, dst))
    # kg.edata['edge_type'] = torch.tensor(edge_type).reshape(-1, 1)
    kg.edata[dgl.EID] = torch.tensor(list(range(kg.num_edges()))).reshape(-1, 1)
    kg.ndata[dgl.NID] = torch.tensor(list(range(kg.num_nodes()))).reshape(-1, 1)
    # kg = dgl.to_bidirected(kg)
    print(kg)

    for file_type, file_path in files.items():
        data = []
        print('processing {} subgraph'.format(file_type))
        with open(file_path, 'r') as f:
            for line in f:
                raw = line.strip().split('\t')
                target, drug, label = int(raw[0]), int(raw[1]), int(raw[2])
                base_target = target
                base_drug = drug
                if id2entity[target] not in entity_label['Gene'] or id2entity[drug] not in entity_label['Compound']:
                    continue

                hetro_target = entity_label['Gene'][id2entity[target]]
                hetro_drug = entity_label['Compound'][id2entity[drug]]
                a_hetro_graph = hetro.in_subgraph({'Gene':[hetro_target], 'Compound':[hetro_drug]})
                node_type_s = {'Compound':[], 'Gene':[], 'Disease':[]}
                for etype in a_hetro_graph.etypes:
                    e = a_hetro_graph.edges(etype=etype)
                    h_type, t_type = etype.split('_')
                    node_type_s[h_type] += e[0].tolist()
                    node_type_s[t_type] += e[1].tolist()
                # h_sg = dgl.node_subgraph(hetro, node_type_s)
                h_sg = []
                for ntype in node_type_s:
                    nids = torch.tensor(hetro_entity_feat[ntype])[node_type_s[ntype]].tolist()
                    if len(nids)!=0:
                        h_sg += nids
                h_sg = set(h_sg[:2*max_num_nodes])
                targets = []
                drugs = []
                for _ in range(hop):
                    a_subgraph = kg.in_subgraph(target, relabel_nodes=True)
                    b_subgraph = kg.in_subgraph(drug, relabel_nodes=True)
                    target = a_subgraph.ndata[dgl.NID]
                    targets += target.tolist()
                    drug = b_subgraph.ndata[dgl.NID]
                    drugs += drug.tolist()
                inter_nodes = set(targets[:max_num_nodes]).union(set(drugs[:max_num_nodes]))
                subgraph = kg.subgraph(torch.tensor(list(inter_nodes.union((base_drug, base_target)))))
                subgraph = dgl.add_self_loop(subgraph)
                h_sg = kg.subgraph(torch.tensor(list(h_sg.union((base_drug, base_target)))))
                h_sg = dgl.add_self_loop(h_sg)
                data.append([base_drug, base_target, label, subgraph, h_sg])
        save_pickle(data, '{}/{}/subgraph/{}/{}_{}_subgraph_{}.pkl'.format(root_path, dataset, file_type, file_type, args.task_type, max_num_nodes))
    
    

    


if __name__=='__main__':
    files = {'train':'dataset/drugbank/dti/random_split/train.tsv',
             'valid':'dataset/drugbank/dti/random_split/valid.tsv',
             'test':'dataset/drugbank/dti/random_split/test.tsv'
             }
    triple_file = 'dataset/drkg/train.tsv'
    # data = load_pickle('dataset/drugbank/subgraph/train/train_subgraph.pkl')
    # # print(data)
    process_files_dti(files, triple_file, max_num_nodes=100, root_path='')
    