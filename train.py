import argparse
import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import pandas as pd

from fake_news_dataset import FakeNewsDataSet
from fake_news_graph_model import FakeNewsGraphModel
from fake_news_mlp_model import FakeNewsMLPModel
from utils.undirected_transformer import UndirectedTransformer
from utils.evaulator import evaluate

@torch.no_grad()
def compute_test(model, loader, device='cpu', multi_gpu=False, verbose=False):
    model.eval()
    loss_test = 0.0
    out_log = []

    for data in loader:
        if not multi_gpu:
            data = data.to(device)

        out = model(data)
        if multi_gpu:
            y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
        else:
            y = data.y

        if verbose:
            print(F.softmax(out, dim=1).cpu().numpy())

        out_log.append([F.softmax(out, dim=1), y])
        loss_test += F.nll_loss(out, y).item()

    evaluation_metrics = evaluate(out_log, loader)
    return evaluation_metrics, loss_test

def run_experiment(dataset_name, feature_type, base_type, model_type=None, 
                   seed=777, epochs=30, batch_size=128, lr=0.01, wd=0.01, hidden_size=128, dropout=0.0, device='cpu'):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    dataset = FakeNewsDataSet(
        root='data',
        feature=feature_type,
        empty=False,
        name=dataset_name,
        transform=UndirectedTransformer()
    )
    num_classes = dataset.num_classes
    num_features = dataset.num_features

    num_training = int(len(dataset) * 0.2)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if base_type.lower() == 'graph':
        model = FakeNewsGraphModel(
            num_features=num_features,
            nhid=hidden_size,
            num_classes=num_classes,
            dropout_ratio=dropout,
            model_type=model_type,
            concat=(feature_type=='bert')
        )
    elif base_type.lower() == 'mlp':
        model = FakeNewsMLPModel(
            num_features=num_features,
            nhid=hidden_size,
            num_classes=num_classes,
            dropout_ratio=dropout,
            pool_type='mean'
        )
    else:
        raise ValueError("Invalid base_type")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        model.train()
        loss_train = 0.0
        out_log = []
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), data.y])

        # Validation step for metric reporting
        [acc_val, _, _, _, _, _, _], val_loss = compute_test(model, val_loader, device=device)
        print(f"Validation-accuracy: {acc_val}")

    [acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(model, test_loader, device=device)
    return {
        'dataset': dataset_name,
        'feature': feature_type,
        'base': base_type,
        'model': model_type if model_type else 'mlp',
        'acc': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'ap': ap
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    configurations = [
        {'dataset': 'politifact', 'model_type': 'gcn', 'feature_type': 'spacy', 'base_type': 'graph'},
        {'dataset': 'politifact', 'model_type': 'gat', 'feature_type': 'bert', 'base_type': 'graph'},
        {'dataset': 'politifact', 'model_type': 'sage', 'feature_type': 'bert', 'base_type': 'graph'},
        {'dataset': 'politifact', 'feature_type': 'bert', 'base_type': 'mlp'}, 

        {'dataset': 'gossipcop', 'model_type': 'gcn', 'feature_type': 'bert', 'base_type': 'graph'},
        {'dataset': 'gossipcop', 'model_type': 'gat', 'feature_type': 'bert', 'base_type': 'graph'},
        {'dataset': 'gossipcop', 'model_type': 'sage', 'feature_type': 'bert', 'base_type': 'graph'},
        {'dataset': 'gossipcop', 'feature_type': 'bert', 'base_type': 'mlp'} 
    ]

    results = []
    for cfg in configurations:
        print(f"Running: {cfg}")
        res = run_experiment(
            dataset_name=cfg['dataset'],
            feature_type=cfg['feature_type'],
            base_type=cfg['base_type'],
            model_type=cfg.get('model_type', None),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            wd=args.wd,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            device=args.device
        )
        results.append(res)

    df = pd.DataFrame(results)
    output_dir = os.path.join('/opt/ml/model', 'results')
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
