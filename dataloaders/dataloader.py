'''
Date: 2023-04-17 22:58:40
LastEditors: xiaomingaaa
LastEditTime: 2023-07-25 16:06:46
FilePath: /debias/dataloaders/dataloader.py
Description: 
'''
import torch
from torch.utils.data import DataLoader, Dataset
from subgraph_util.graph_utils import load_pickle
from chem_utils import get_protein_des, get_smiles_ECPF
import json
import numpy as np
import logging

class RandomData(Dataset):
    def __init__(self, args, file_type='train', KG=False, k_fold=False) -> None:
        self.samples = []
        print('processing dataset')
        if k_fold:
            if KG:
                print('KGE ', KG)
                entity_embed = np.load('ckpts/pretrained_embed/DRKG_TransE_entity.npy')
                for file_type in ['train', 'valid', 'test']:
                    data = load_pickle('{}/{}/subgraph/{}/{}_subgraph_{}.pkl'.format(args.root_dir, args.dataset, file_type, file_type, args.task_type, args.max_num_nodes))
                    for (did, pid, label, subgraph, h_sg) in data:
                        did, pid = str(did), str(pid)

                        p_feat = np.concatenate([entity_embed[int(pid)]], axis=-1)
                        drug_fp = np.concatenate([entity_embed[int(did)]], axis=-1)
                        self.samples.append([p_feat, drug_fp, int(label), subgraph, h_sg])
            else:
                print('KGE ', KG)
                drug_info = json.load(open('dataset/{}/dti/drug_feat_info.json'.format(args.dataset), 'r'))
                target_info = json.load(open('dataset/{}/dti/target_feat_info.json'.format(args.dataset),'r'))

                for file_type in ['train', 'valid', 'test']:
                    data = load_pickle('{}/{}/subgraph/{}/{}_subgraph_{}.pkl'.format(args.root_dir, args.dataset, file_type, file_type, args.max_num_nodes))
                    for (did, pid, label, subgraph, h_sg) in data:
                        did, pid = str(did), str(pid)
                        if did not in drug_info:
                            continue
                        p_feat = np.array(target_info[pid])
                        drug_fp = np.array(drug_info[did])
                        # drug_fp = get_smiles_ECPF(smiles)
                        # p_feat = get_protein_des(p_seq)
                        self.samples.append([p_feat, drug_fp, int(label), subgraph, h_sg])

        else:
            if KG:
                print('KGE ', KG)
                entity_embed = np.load('ckpts/pretrained_embed/DRKG_TransE_entity.npy')
                path = '{}/{}/subgraph/{}/{}_{}_subgraph_{}.pkl'.format(args.root_dir, args.dataset, file_type, file_type, args.task_type, args.max_num_nodes)

                data = load_pickle(path)
                for (did, pid, label, subgraph, h_sg) in data:
                    did, pid = str(did), str(pid)
                    
                    p_feat = np.concatenate([entity_embed[int(pid)]], axis=-1)
                    drug_fp = np.concatenate([entity_embed[int(did)]], axis=-1)
                    self.samples.append([p_feat, drug_fp, int(label), subgraph, h_sg])
            else:
                print('KGE ', KG)
                drug_info = json.load(open('dataset/{}/dti/drug_feat_info.json'.format(args.dataset), 'r'))
                target_info = json.load(open('dataset/{}/dti/target_feat_info.json'.format(args.dataset),'r'))
                data = load_pickle('{}/{}/subgraph/{}/{}_subgraph_{}.pkl'.format(args.root_dir, args.dataset, file_type, file_type, args.max_num_nodes))
                for (did, pid, label, subgraph, h_sg) in data:
                    did, pid = str(did), str(pid)
                    if did not in drug_info:
                        continue
                    p_feat = np.array(target_info[pid])
                    drug_fp = np.array(drug_info[did])
                    # drug_fp = get_smiles_ECPF(smiles)
                    # p_feat = get_protein_des(p_seq)
                    self.samples.append([p_feat, drug_fp, int(label), subgraph, h_sg])
        
    def __len__(self):
        l = len(self.samples)
        return l

    def __getitem__(self, index):
        feature = self.samples[index]
        return feature