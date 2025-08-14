"""
GNN pipeline for composition -> property regression
-------------------------------------------------
This script implements the pipeline described in the paper you provided,
adapted to your Perovskite_data.csv dataset.

Key features:
- Parse chemical compositions and create small graphs (center = element with
  highest atomic fraction; edges from center to all other elements).
- Node features: [atomic_fraction, atomic_mass]
- Models: GCN and GraphSAGE regression variants using PyTorch Geometric
- Training: configurable splits, early stopping, checkpoint saving
- Explainability: basic GNNExplainer integration for node/edge importance

Note: This script ONLY *implements* and prepares training. It does not attempt
to run in this environment. Run it locally on your machine with CUDA (your
RTX 4060) where PyTorch + PyTorch Geometric are installed.

Usage example (after installing deps):
    python gnn_perovskite_pipeline.py \
        --data /path/to/Perovskite_data.csv \
        --target "band_gap (eV)" \
        --model gcn \
        --epochs 500 \
        --batch_size 32 \
        --out_dir ./outputs

Environment setup (recommended):
    # install matching torch + cudatoolkit for your CUDA version (example for CUDA 11.8)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    # install PyTorch Geometric - follow official instructions for your torch/cuda:
    pip install "torch-geometric" "torch-scatter" "torch-sparse" "torch-cluster" "torch-spline-conv"
    pip install scikit-learn pandas tqdm

If you hit `libnccl.so.2` or similar NCCL errors, common fixes:
- Make sure your locally-installed PyTorch matches the system CUDA driver version.
- Ensure that `LD_LIBRARY_PATH` contains the path to CUDA libs (e.g. /usr/local/cuda/lib64).
- Reinstall PyTorch wheel that matches CUDA (see https://pytorch.org).

"""

import os
import re
import argparse
import math
from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

# torch_geometric imports
try:
    from torch_geometric.data import Data, InMemoryDataset, DataLoader
    from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
    from torch_geometric.explain import GNNExplainer
except Exception as e:
    raise ImportError("torch_geometric is required. Install it following https://pytorch-geometric.readthedocs.io/ ") from e

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---- atomic masses table (extendable) ----
ATOMIC_MASSES = {
 'H':1.0079,'He':4.0026,'Li':6.941,'Be':9.0122,'B':10.811,'C':12.0107,'N':14.0067,'O':15.999,'F':18.9984,'Ne':20.1797,
 'Na':22.9897,'Mg':24.305,'Al':26.9815,'Si':28.0855,'P':30.9738,'S':32.065,'Cl':35.453,'Ar':39.948,'K':39.0983,'Ca':40.078,
 'Sc':44.9559,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938,'Fe':55.845,'Co':58.9332,'Ni':58.6934,'Cu':63.546,'Zn':65.39,
 'Ga':69.723,'Ge':72.64,'As':74.9216,'Se':78.96,'Br':79.904,'Kr':83.8,'Rb':85.4678,'Sr':87.62,'Y':88.9059,'Zr':91.224,
 'Nb':92.9064,'Mo':95.94,'Tc':98,'Ru':101.07,'Rh':102.9055,'Pd':106.42,'Ag':107.8682,'Cd':112.411,'In':114.818,'Sn':118.71,
 'Sb':121.76,'Te':127.6,'I':126.9045,'Xe':131.293,'Cs':132.9054,'Ba':137.327,'La':138.9055,'Ce':140.116,'Pr':140.9077,'Nd':144.24,
 'Pm':145,'Sm':150.36,'Eu':151.964,'Gd':157.25,'Tb':158.9253,'Dy':162.5,'Ho':164.9303,'Er':167.259,'Tm':168.9342,'Yb':173.04,
 'Lu':174.967,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.9665,'Hg':200.59,
 'Tl':204.3833,'Pb':207.2,'Bi':208.9804,'Po':209,'At':210,'Rn':222,'Fr':223,'Ra':226,'Ac':227,'Th':232.0381,'Pa':231.0359,'U':238.0289
}

# ---- composition parser ----
ELEMENT_RE = re.compile(r'([A-Z][a-z]?)([0-9]*\.?[0-9]*)')


def parse_composition(formula: str) -> Dict[str, float]:
    """Parse a chemical formula into a dict element->stoich amount.
    This is a simple parser handling formulas like 'AB3', 'Cs0.5MA0.5PbI3', 'La2 Ni O4'.
    It does not fully support nested parentheses or complex charge notations.
    Returns a dict or empty dict if parsing fails.
    """
    if not isinstance(formula, str):
        return {}
    s = formula.replace(' ', '')
    # strip hydration dot part
    s = s.split('·')[0]
    matches = ELEMENT_RE.findall(s)
    if not matches:
        return {}
    counts = defaultdict(float)
    for el, num in matches:
        if num == '':
            numf = 1.0
        else:
            try:
                numf = float(num)
            except:
                numf = float(num.replace(',', '.'))
        counts[el] += numf
    return dict(counts)


# ---- graph building rule ----
def build_graph_from_composition(comp: Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Return node feature tensor (N,2), edge_index tensor (2,E), element list.
    Node features: [fraction, atomic_mass]
    Graph rule: center = element with highest fraction; connect center to all others (undirected);
    self-loops added automatically if desired by Conv layers (we keep no explicit self-loop here).
    """
    if not comp:
        return None
    elems = list(comp.keys())
    amounts = np.array([comp[e] for e in elems], dtype=float)
    fracs = amounts / amounts.sum()
    n = len(elems)
    feats = np.zeros((n, 2), dtype=np.float32)
    for i, e in enumerate(elems):
        feats[i, 0] = float(fracs[i])
        feats[i, 1] = float(ATOMIC_MASSES.get(e, 0.0))
    # build edges
    center_idx = int(np.argmax(fracs))
    edge_list = []
    for j in range(n):
        if j == center_idx:
            continue
        edge_list.append((center_idx, j))
        edge_list.append((j, center_idx))
    if len(edge_list) == 0:
        # single element compound - self-loop
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    x = torch.from_numpy(feats)
    return x, edge_index, elems


# ---- PyG Dataset wrapper ----
class PerovskiteGNNDataset(InMemoryDataset):
    def __init__(self, csv_path: str, target_col: str, comp_col_candidates: List[str]=None, transform=None):
        super().__init__(None, transform)
        self.csv_path = csv_path
        self.target_col = target_col
        self.data_list = []
        self._prepare(csv_path, comp_col_candidates)
        self.data, self.slices = self.collate(self.data_list)

    def _find_composition_column(self, df: pd.DataFrame, candidates=None):
        if candidates is None:
            candidates = ['composition', 'formula', 'formula_pretty', 'pretty_formula']
        for c in candidates:
            if c in df.columns:
                return c
        # fallback heuristic
        for c in df.columns:
            if df[c].dtype == object and df[c].str.contains('[A-Z]', regex=True).any():
                return c
        return None

    def _prepare(self, csv_path: str, comp_col_candidates: List[str]=None):
        df = pd.read_csv(csv_path)
        if self.target_col not in df.columns:
            raise ValueError(f"Target column {self.target_col} not found in CSV. Columns: {df.columns.tolist()}")
        comp_col = self._find_composition_column(df, comp_col_candidates)
        if comp_col is None:
            raise ValueError("Could not automatically find a composition column. Pass comp_col_candidates to constructor.")
        usable = 0
        for idx, row in df.iterrows():
            formula = row[comp_col]
            comp = parse_composition(formula)
            if not comp:
                continue
            graph = build_graph_from_composition(comp)
            if graph is None:
                continue
            x, edge_index, elems = graph
            # ensure numeric target
            try:
                y = float(row[self.target_col])
            except:
                continue
            data = Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.float))
            data.formula = str(formula)
            data.elems = elems
            self.data_list.append(data)
            usable += 1
        if usable == 0:
            raise RuntimeError("No usable rows found after parsing compositions and targets. Check CSV and column names.")
        print(f"Prepared {usable} graphs from {csv_path} (comp_col={comp_col})")


# ---- GNN models ----
class GCNRegressor(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Linear(hidden_channels//2, 1)
        )

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        hg = global_mean_pool(h, batch)  # (batch_size, hidden)
        return self.head(hg).squeeze(-1)


class SAGERegressor(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers-1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Linear(hidden_channels//2, 1)
        )

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        hg = global_mean_pool(h, batch)
        return self.head(hg).squeeze(-1)


# ---- training / evaluation utilities ----

def train_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def eval_dataset(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds.append(out.cpu().numpy())
            trues.append(data.y.view(-1).cpu().numpy())
    if len(preds) == 0:
        return None
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    rmse = mean_squared_error(trues, preds) ** 0.5
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds) if len(trues) > 1 else float('nan')
    return {'preds': preds, 'trues': trues, 'rmse': rmse, 'mae': mae, 'r2': r2}


# ---- main orchestration ----

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print('Using device:', device)

    dataset = PerovskiteGNNDataset(args.data, args.target)

    # train/val/test split (paper uses 80/10/10 for one task) - configurable here
    n = len(dataset)
    idxs = list(range(n))
    trainval_idx, test_idx = train_test_split(idxs, test_size=args.test_frac, random_state=42)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=args.val_frac/(1-args.test_frac), random_state=42)

    train_ds = dataset[train_idx[0]:train_idx[-1]+1] if False else torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # model selection
    if args.model.lower() == 'gcn':
        model = GCNRegressor(in_channels=2, hidden_channels=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)
    else:
        model = SAGERegressor(in_channels=2, hidden_channels=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float('inf')
    best_state = None
    patience = args.patience
    cur_pat = 0

    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_res = eval_dataset(model, val_loader, device)
        if val_res is not None:
            val_rmse = val_res['rmse']
            val_mae = val_res['mae']
            val_r2 = val_res['r2']
        else:
            val_rmse = float('nan')
        print(f"Epoch {epoch:04d} | Train loss: {train_loss:.6f} | Val RMSE: {val_rmse:.6f}")
        if val_res is not None and val_rmse < best_val:
            best_val = val_rmse
            best_state = model.state_dict()
            cur_pat = 0
            torch.save(best_state, os.path.join(args.out_dir, f"best_{args.model}.pth"))
        else:
            cur_pat += 1
        if cur_pat >= patience:
            print(f"Early stopping at epoch {epoch} (patience {patience})")
            break

    # Load best model and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    test_res = eval_dataset(model, test_loader, device)
    if test_res is not None:
        print("Test RMSE: {:.4f} | MAE: {:.4f} | R2: {:.4f}".format(test_res['rmse'], test_res['mae'], test_res['r2']))

    # Save final model and predictions
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"final_{args.model}.pth"))
    preds_df = pd.DataFrame({'true': test_res['trues'], 'pred': test_res['preds']})
    preds_df.to_csv(os.path.join(args.out_dir, f"preds_{args.model}.csv"), index=False)

    # Explainability: GNNExplainer on a few test graphs
    explainer = GNNExplainer(model, epochs=args.explain_epochs)
    explain_outs = []
    # limit number of examples
    n_explain = min(len(test_ds), args.max_explain)
    for i in range(n_explain):
        data = test_ds[i]
        data = data.to(device)
        node_imp, edge_imp = explainer.explain_graph(data.x, data.edge_index)
        explain_outs.append({'formula': getattr(data, 'formula', None), 'node_imp': node_imp.cpu().numpy(), 'edge_imp': edge_imp.cpu().numpy()})
    # save explanations
    import json
    expl_path = os.path.join(args.out_dir, f"explanations_{args.model}.json")
    with open(expl_path, 'w') as f:
        json.dump(explain_outs, f, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f"Saved explanations to {expl_path}")


# ---- argparser ----
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help='Path to Perovskite_data.csv')
    p.add_argument('--target', type=str, default='band_gap (eV)', help='Target column name in CSV')
    p.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage'], help='Model type')
    p.add_argument('--out_dir', type=str, default='./outputs', help='Directory to save models and outputs')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-6)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--test_frac', type=float, default=0.1)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    p.add_argument('--explain_epochs', type=int, default=200)
    p.add_argument('--max_explain', type=int, default=10)

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
