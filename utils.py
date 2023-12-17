from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score,precision_recall_curve,auc, f1_score,cohen_kappa_score
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from tqdm import tqdm
from dataloaders.dataloader import RandomData
from torch.utils.data import DataLoader, Subset

def evaluate(y_pred, labels):
    roc_auc = roc_auc_score(labels, y_pred)
    pr,re,_=precision_recall_curve(labels,y_pred,pos_label=1)
    aupr = auc(re, pr)
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    recall = recall_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    acc = accuracy_score(labels, y_pred)
    return roc_auc, recall, precision, acc, aupr

def evaluate_multi_class(y_pred, labels):
    # import ipdb; ipdb.set_trace()
    # roc_auc = roc_auc_score(labels, y_pred)
    
    # y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    scores = np.max(y_pred, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    auc = f1_score(labels, y_pred, average='macro')
    aupr = f1_score(labels, y_pred, average='micro')
    kappa = cohen_kappa_score(labels, y_pred)
    recall = recall_score(labels, y_pred, average='macro')
    # precision = precision_score(labels, y_pred)
    acc = accuracy_score(labels, y_pred)
    return auc, recall, kappa, acc, aupr

def eval_loader(model, loader, device, mc=False):
    if mc:
        e = evaluate_multi_class
    else:
        e = evaluate
    model.eval()
    # model.cuda()
    model = model.to(device)
    Y_pre = []
    Y_true = []
    pre_loss = []
    bar = tqdm(enumerate(loader))
    for b_idx, batch in bar:
        (d, p, subgraph, h_sg), l = batch
        p = p.float().to(device)
        d = d.float().to(device)
        l = l.float().to(device)
        # subgraph = subgraph.to(device)
        h_sg = h_sg.to(device)
        # data = data.to(device)
        with torch.no_grad():
            loss, pred, label = model(p, d, l, subgraph, h_sg)
            pre_loss.append(loss.cpu().detach().numpy())
            Y_pre.extend(list(pred.cpu().detach().numpy()))
            Y_true.extend(list(label.cpu().detach().numpy()))
        bar.set_description('Evaluating: {}/{}'.format(str(b_idx+1), len(loader)))
    return np.mean(pre_loss), e(np.array(Y_pre), np.array(Y_true))



def load_data(args):
    if args.k_fold:
        dataset = RandomData(args, KG=args.KG, k_fold=args.k_fold)
        frac = 1/args.k
        total_size = len(dataset)
        seg = int(frac*total_size)
        for i in range(args.k-1):
            trll = 0
            trlr = i * seg
            vall = trlr
            valr = vall + seg
            tsll = valr
            tslr = tsll + seg
            trrl = tslr
            trrr = total_size

            train_indices = list(range(trll,trlr)) + list(range(trrl,trrr))
            val_indices = list(range(vall,valr))
            test_indices = list(range(tsll,tslr))

            train_set = Subset(dataset, train_indices)
            train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=args.collate_fn)
            val_set = Subset(dataset, val_indices)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=args.collate_fn)
            test_set = Subset(dataset, test_indices)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=args.collate_fn)
            loader = [train_loader, val_loader, test_loader]
            yield loader
    else:
        dataset = RandomData(args, file_type='train', KG=args.KG)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=args.collate_fn)
        valid = RandomData(args, file_type='valid', KG=args.KG)
        valid_loader = DataLoader(valid, batch_size=args.batch_size, collate_fn=args.collate_fn)
        test_data  = RandomData(args, file_type='test', KG=args.KG)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=args.collate_fn)
        loader = [train_loader, valid_loader, test_loader]
        yield loader

def load_data_k_fold(args, k):
    dataset = RandomData(args, KG=args.KG, k_fold=True)
    frac = 1/k
    seg = int(frac*len(dataset))
    loaders = []
    for i in range(k-1):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = vall + seg
        tsll = valr
        tslr = tsll + seg
        trrl = tslr
        trrr = len(dataset)

        train_indices = list(range(trll,trlr)) + list(range(trrl,trrr))
        val_indices = list(range(vall,valr))
        test_indices = list(range(tsll,tslr))
        
        train_set = Subset(dataset, train_indices)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=args.collate_fn)
        val_set = Subset(dataset, val_indices)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=args.collate_fn)
        test_set = Subset(dataset, test_indices)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=args.collate_fn)

        yield train_loader, val_loader, test_loader

    
    

def save_model(args, model):
    path = '{}/ds_{}_num_{}_dim{}_{}.pt'.format(args.ckpt_dir, args.dataset, args.max_num_nodes, args.embed_dim, args.flag)

    torch.save(model.state_dict(), path)

def load_model(args, model):
    path = '{}/ds_{}_num_{}_dim{}_{}.pt'.format(args.ckpt_dir, args.dataset, args.max_num_nodes, args.embed_dim, args.flag)
    model.load_state_dict(torch.load(path))
    return model

def save_gnn_model(args, model):
    path = 'ckpts/gnn.pt'

    torch.save(model.state_dict(), path)

def load_gnn_model(args, model):
    path = 'ckpts/gnn.pt'
    model.load_state_dict(torch.load(path))
    return model


def log_info(args, info):   
    args.log_file.write('{}\n'.format(info))

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
