"""
GNN Model — Message Passing Neural Network for molecular property prediction.

Uses PyTorch + PyTorch Geometric to operate directly on molecular graphs.
Provides a deep learning baseline for comparison with XGBoost.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from torch_geometric.data import Data, Dataset, DataLoader as PyGDataLoader
    from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, BatchNorm
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import r2_score, mean_absolute_error
from src.config import SUBTYPES

logger = logging.getLogger(__name__)

if not HAS_PYG:
    logger.warning("torch-geometric not installed. GNN model will not be available.")


# SUBTYPES imported from src.config

# ── Atom & Bond Featurization ──────────────────────────────────

ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "num_hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
}


def _one_hot(value, choices):
    """One-hot encode a value from a list of choices."""
    encoding = [0] * (len(choices) + 1)  # +1 for unknown
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1  # unknown
    return encoding


def atom_features(atom) -> list:
    """Compute atom feature vector (dimension: ~140)."""
    features = []
    features.extend(_one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"]))
    features.extend(_one_hot(atom.GetDegree(), ATOM_FEATURES["degree"]))
    features.extend(_one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"]))
    features.extend(_one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"]))
    features.extend(_one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"]))
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    return features


def bond_features(bond) -> list:
    """Compute bond feature vector (dimension: ~7)."""
    features = []
    features.extend(_one_hot(bond.GetBondType(), BOND_FEATURES["bond_type"]))
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    return features


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert a SMILES string to a PyTorch Geometric Data object."""
    if not HAS_PYG:
        raise ImportError("torch-geometric is required for GNN model")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node (atom) features
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    # Edge (bond) features — bidirectional
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bf, bf])

    if len(edge_index) == 0:
        # Single atom molecule — add self-loop
        edge_index = [[0, 0]]
        edge_attr = [[0] * BOND_DIM]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ── GNN Architecture ───────────────────────────────────────────

class MoleculeGNN(nn.Module):
    """
    Message Passing Neural Network for molecular property prediction.
    
    Architecture:
    - 3 GINE (Graph Isomorphism Network with Edge features) layers
    - BatchNorm after each layer
    - Global mean+max pooling concatenation
    - 2-layer MLP prediction head with dropout
    """

    def __init__(self, node_dim: int = ATOM_DIM, edge_dim: int = BOND_DIM, 
                 hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINEConv(mlp, edge_dim=edge_dim)
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden_dim))
        
        # Edge encoder to match hidden_dim
        self.edge_encoder = nn.Linear(edge_dim, edge_dim)
        
        # Prediction head: global pool → MLP
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling concat
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Encode nodes
        x = self.node_encoder(x)
        
        # Message passing layers
        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # Residual connection
        
        # Global pooling (mean + max concatenation)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pool = torch.cat([x_mean, x_max], dim=1)
        
        # Prediction
        out = self.head(x_pool)
        return out.squeeze(-1)


# ── Training & Evaluation ─────────────────────────────────────

def _prepare_data(subtype: str, data_path: str = "data/raw"):
    """Load data, scaffold-split, and convert to PyG graphs."""
    from src.data_loader import load_and_clean
    from src.scaffold_split import split_smiles_globally
    
    df, _ = load_and_clean(data_path, mode="precise", 
                            save_lookup_path="data/processed/db_lookup_gnn_temp.json",
                            include_decoys=False)
    df_st = df[df["target_subtype"] == subtype].copy().reset_index(drop=True)
    
    if len(df_st) < 50:
        raise ValueError(f"Insufficient data for {subtype} ({len(df_st)} samples)")
    
    import json
    from pathlib import Path
    
    # Enforce Global Scaffold Split
    split_path = Path("data/processed/global_split.json")
    if not split_path.exists():
        raise FileNotFoundError("Global split not found! Run retrain_production.py first.")
        
    with open(split_path, "r") as f:
        global_split = json.load(f)
        
    train_smiles_set = set(global_split["train"])
    test_smiles_set = set(global_split["test"])
    
    train_df = df_st[df_st["canonical_smiles"].isin(train_smiles_set)].reset_index(drop=True)
    test_df = df_st[df_st["canonical_smiles"].isin(test_smiles_set)].reset_index(drop=True)
    
    # Convert to graphs
    def df_to_graphs(df_sub):
        graphs = []
        for _, row in df_sub.iterrows():
            g = smiles_to_graph(row["canonical_smiles"])
            if g is not None:
                g.y = torch.tensor([row["pchembl_value"]], dtype=torch.float)
                g.smiles = row["canonical_smiles"]
                graphs.append(g)
        return graphs
    
    train_graphs = df_to_graphs(train_df)
    test_graphs = df_to_graphs(test_df)
    
    logger.info("[GNN] %s: Train=%d, Test=%d molecular graphs", subtype, len(train_graphs), len(test_graphs))
    
    return train_graphs, test_graphs


def train_gnn_model(subtype: str, epochs: int = 100, lr: float = 1e-3, batch_size: int = 64,
                    patience: int = 15, data_path: str = "data/raw"):
    """Train a GNN model for a specific receptor subtype."""
    if not HAS_PYG:
        logger.error("torch-geometric not installed. Cannot train GNN model.")
        return None

    logger.info("=" * 60)
    logger.info("GNN TRAINING FOR %s (MPNN/GINE, epochs=%d)", subtype, epochs)
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  Device: %s", device)
    
    # Prepare data
    train_graphs, test_graphs = _prepare_data(subtype, data_path)
    
    train_loader = PyGDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = PyGDataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = MoleculeGNN(node_dim=ATOM_DIM, edge_dim=BOND_DIM, hidden_dim=256, num_layers=3, dropout=0.2)
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    loss_fn = nn.MSELoss()
    
    best_val_mae = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)
        
        # Evaluate
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch)
                all_preds.extend(pred.cpu().numpy().tolist())
                all_true.extend(batch.y.cpu().numpy().tolist())
        
        val_mae = float(mean_absolute_error(all_true, all_preds))
        val_r2 = float(r2_score(all_true, all_preds))
        
        scheduler.step(val_mae)
        
        if epoch % 10 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info("  Epoch %3d/%d | Train Loss: %.4f | Val MAE: %.4f | Val R²: %.4f | LR: %.1e",
                         epoch, epochs, train_loss, val_mae, val_r2, current_lr)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_no_improve = 0
            # Save best model
            model_dir = Path("models/gnn")
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "node_dim": ATOM_DIM,
                "edge_dim": BOND_DIM,
                "hidden_dim": 256,
                "num_layers": 3,
                "best_val_mae": best_val_mae,
                "best_epoch": best_epoch,
            }, model_dir / f"gnn_{subtype.lower()}_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("  [EARLY STOP] No improvement for %d epochs. Best epoch: %d (MAE: %.4f)",
                             patience, best_epoch, best_val_mae)
                break

    # Final evaluation with best model
    # Note: weights_only=False is used because checkpoints store custom metadata and python dicts.
    # Safe since checkpoints are produced locally by this pipeline, not fetched from untrusted sources.
    checkpoint = torch.load(Path("models/gnn") / f"gnn_{subtype.lower()}_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_true.extend(batch.y.cpu().numpy().tolist())
    
    final_mae = float(mean_absolute_error(all_true, all_preds))
    final_r2 = float(r2_score(all_true, all_preds))
    final_rmse = float(np.sqrt(np.mean((np.array(all_true) - np.array(all_preds))**2)))

    logger.info("[FINAL] %s GNN Performance: R²=%.4f, MAE=%.4f, RMSE=%.4f",
                 subtype, final_r2, final_mae, final_rmse)
    logger.info("    Best Epoch = %d", best_epoch)
    
    result = {
        "subtype": subtype,
        "model": "MPNN/GINE",
        "r2": final_r2,
        "mae": final_mae,
        "rmse": final_rmse,
        "best_epoch": best_epoch,
        "train_size": len(train_graphs),
        "test_size": len(test_graphs),
    }
    
    # Save report
    report_dir = Path("outputs/gnn")
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / f"{subtype}_gnn_report.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result


def train_all_subtypes(epochs: int = 100):
    """Train GNN models for ALL 4 receptor subtypes."""
    logger.info("=" * 60)
    logger.info("GNN TRAINING FOR ALL SUBTYPES")
    logger.info("=" * 60)

    all_results = {}
    for st in SUBTYPES:
        result = train_gnn_model(subtype=st, epochs=epochs)
        if result is not None:
            all_results[st] = result

    report_dir = Path("outputs/gnn")
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": "MPNN/GINE (PyTorch Geometric)",
        "n_subtypes": len(all_results),
        "results": all_results,
    }
    with open(report_dir / "all_subtypes_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("All GNN results saved to %s", report_dir / "all_subtypes_summary.json")

    return all_results


def predict_gnn(smiles: str, subtype: str) -> Optional[float]:
    """Run GNN inference on a single SMILES for a specific subtype."""
    if not HAS_PYG:
        return None
    
    model_path = Path(f"models/gnn/gnn_{subtype.lower()}_model.pt")
    if not model_path.exists():
        return None
    
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None
    
    device = torch.device("cpu")
    # Safe to use weights_only=False here as checkpoints are generated locally by our pipeline.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = MoleculeGNN(
        node_dim=checkpoint.get("node_dim", ATOM_DIM),
        edge_dim=checkpoint.get("edge_dim", BOND_DIM),
        hidden_dim=checkpoint.get("hidden_dim", 256),
        num_layers=checkpoint.get("num_layers", 3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Add batch dimension
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    
    with torch.no_grad():
        pred = model(graph)
    
    return float(pred.item())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GNN (MPNN) model training for adenosine receptor subtypes")
    parser.add_argument("--subtype", default=None, help="Subtype to train")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--all", action="store_true", help="Train ALL 4 subtypes")
    args = parser.parse_args()
    
    if args.all:
        train_all_subtypes(epochs=args.epochs)
    elif args.subtype:
        train_gnn_model(subtype=args.subtype, epochs=args.epochs)
    else:
        train_all_subtypes(epochs=args.epochs)
