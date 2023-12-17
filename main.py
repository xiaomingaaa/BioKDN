from config.model_config import model_config
import torch
import numpy as np
import time
import os
from utils import evaluate, eval_loader, load_data, save_model, load_model, log_info, get_parameter_number
import torch.nn.functional as F
from models.layers import Regularization

from torch.optim import Adam, SGD
from models.model import GraphClassifier
from subgraph_util.graph_utils import process_files_dti, collate_dgl, load_kge_embed
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(6789)
np.random.seed(6789)
torch.cuda.manual_seed_all(6789)
os.environ['PYTHONHASHSEED'] = str(6789)

def main(args):
    model_setting = model_config(KG=args.KG)
    loader_gen = load_data(args)
    results = []
    for idx, loader in enumerate(loader_gen):
        print('++++++++++{}-Fold Training+++++++++++'.format(idx))
        train_loader, valid_loader, test_loader = loader
        model = GraphClassifier(args, model_setting.drug_hidden_dim, model_setting.protein_hidden_dim)
        print('Number of Parameters: ', get_parameter_number(model))
        reg = Regularization(model, args.weight_decay)
        if args.gpu:
            model.cuda()
            reg.cuda()
        optim = Adam(model.parameters(), lr=args.lr)
        early_stop = 0
        best_roc = 0.
        for i in range(model_setting.epoch):
            model.train()
            loss_t = 0
            bar = tqdm(enumerate(train_loader))
            for b_idx, batch in bar:
                (drug, target, subgraph, h_sg), label = batch
                target = target.float()
                drug = drug.float()
                label = label.float()
                if args.gpu:
                    target = target.cuda()
                    drug = drug.cuda()
                    label = label.cuda()
                    # subgraph = subgraph.to(args.device)
                    h_sg = h_sg.to(args.device)

                optim.zero_grad()
                loss, logits, label = model(target, drug, label, subgraph, h_sg)
                # loss += reg(model)
                loss.backward()
                optim.step()

                loss_t+=loss.detach().cpu().item()
                bar.set_description('Training: epoch-{} |'.format(i+1) + str(b_idx+1) + '/{} loss_train: '.format(len(train_loader)) + str(loss.cpu().detach().numpy()))
            loss_print = loss_t/b_idx
            early_stop+=1
            _, (roc_auc, recall, precision, acc, aupr) = eval_loader(model, valid_loader, args.device, mc = args.split_type=='multi_class')
            print('Epoch: {} | train_loss: {}, auc: {}, aupr: {}, recall: {}, precision: {}, acc: {}'.format(i+1, loss_print, roc_auc, aupr, recall, precision, acc))
            if roc_auc > best_roc:
                best_roc = roc_auc
                save_model(args, model)
                print('best model saved!!!')
                log_info(args, 'Best model saved, Epoch: {} | train_loss: {}, auc: {}, recall: {}, precision: {}, acc: {}'.format(i+1, loss_print, roc_auc, recall, precision, acc))
                early_stop = 0
            if early_stop > 20:
                break
        model = load_model(args, model)
        eval_loss, (roc_auc, recall, precision, acc, aupr) = eval_loader(model, test_loader, args.device, mc = args.split_type=='multi_class') 
        print('Test_loss: {}, auc: {}, aupr: {}, recall: {}, precision: {}, acc: {}'.format(eval_loss, roc_auc, aupr, recall, precision, acc))
        log_info(args, 'Test_loss: {}, auc: {}, aupr: {}, recall: {}, precision: {}, acc: {}'.format(eval_loss, roc_auc, aupr, recall, precision, acc))
        results.append([roc_auc, recall, precision, acc, aupr])
    log_info(args, str(results))
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='gpu id')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate of pretrain')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--split_type', type=str, default='random_split',
                        help='dataset')
    parser.add_argument('--KG', action='store_true',
                        help='is kg')
    parser.add_argument('--root_dir', type=str, default='/root/autodl-tmp',
                        help='the saved root path of subgraph')
    
    parser.add_argument('--max_num_nodes', type=int, default=200,
                        help='the max number of neighbor of subgraph')
    
    ### Graph Structure Learning Parameter
    parser.add_argument('--num_pers', type=int, default=2, 
                        help='The layer num for graph learner')
    parser.add_argument('--input_size', type=int, default=100, 
                        help='The input feature size of graph learner')
    parser.add_argument('--hidden_size', type=int, default=32, 
                        help='The hidden size of graph learner')
    parser.add_argument('--metric_type', type=str, default='attention', 
                        help='The sim function of graph learner')
    parser.add_argument('--top_k', type=int, default=10, 
                        help='The sim function of graph learner')
    parser.add_argument("--graph_type", type=str, default="prob", 
                        help="epsilonNN, KNN, prob")
    parser.add_argument("--feature_denoise", type=bool, default=True, 
                        help="epsilonNN, KNN, prob")
    parser.add_argument("--kge", type=bool, default=True, 
                        help="epsilonNN, KNN, prob")
    
    ### graph classifier
    parser.add_argument('--embed_dim', type=int, default=100, 
                        help='The input feature size of graph learner')
    parser.add_argument('--task_type', type=str, default='dti', 
                        help='dti or ddi')
    
    # model save
    parser.add_argument('--ckpt_dir', type=str, default='ckpts/model',
                        help='the saved root path of subgraph')
    parser.add_argument('--dataset', type=str, default='drugbank',
                        help='the saved root path of subgraph')
    parser.add_argument('--log', type=bool, default=True,
                        help='the saved root path of subgraph')
    parser.add_argument('--k_fold', type=bool, default=False,
                        help='used k fold split')
    parser.add_argument('--k', type=int, default=10,
                        help='k fold')
    parser.add_argument('--flag', type=str, default='test',
                        help='k fold')
    parser.add_argument('--weight_decay', type=float, default=0.0017,
                        help='')

    args = parser.parse_args()
    print(args)
    args.collate_fn = collate_dgl
    if args.kge:
        args.load_kge = load_kge_embed
    
    if args.log:
        log_path = 'logs/{}/log_num{}_bs{}_lr{}_{}_{}_{}_{}.txt'.format(args.dataset, args.max_num_nodes, args.batch_size
                                                              , args.lr, args.split_type, args.embed_dim, args.KG, args.flag)
        args.log_file = open(log_path, 'w')
        
    files = {'train':'dataset/{}/{}/{}/train.tsv'.format(args.dataset, args.task_type, args.split_type),
             'valid':'dataset/{}/{}/{}/valid.tsv'.format(args.dataset, args.task_type, args.split_type),
             'test':'dataset/{}/{}/{}/test.tsv'.format(args.dataset, args.task_type, args.split_type)
             }
    # triple_file = 'dataset/drkg/train.tsv'
    triple_file = '/root/autodl-fs/DTI/drkg/noisy_kg/add_semantic_0_5.tsv'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_file = '{}/{}/subgraph/train/train_{}_subgraph_{}.pkl'.format(args.root_dir, args.dataset, args.task_type, args.max_num_nodes)
    print(root_file)
    if not os.path.exists(root_file):
        process_files_dti(files, triple_file, args.dataset, args.max_num_nodes, args.root_dir, task_type=args.task_type)
    
    main(args)
    
    if args.log:
        args.log_file.close()