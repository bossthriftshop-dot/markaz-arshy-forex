"""
gng_model.py (ADVANCED, FULL VERSION)
=====================================

Implementasi Growing Neural Gas (GNG) trading dengan dukungan fitur multi-dimensi,
integrasi technical_indicators.py refactor, retrain, dan statistik min/max.
Tetap support pipeline lama & baru (backward compatible).

Copyright (c) 2024.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd

from technical_indicators import (
    extract_features_full,
    detect_structure,
    detect_order_blocks_multi,
    detect_fvg_multi,
    detect_engulfing,
    detect_pinbar,
    get_daily_high_low,
    get_pivot_points,
)

# ==================== GNG MODEL ====================

class GrowingNeuralGas:
    """
    Growing Neural Gas (GNG) dengan update node/edge, training, dan scoring fitur.
    Mendukung pipeline multi-dimensi (AI/ML-ready).
    """

    def __init__(self, max_nodes: int = 100, input_dim: int = 1):
        self.max_nodes = max_nodes
        self.input_dim = input_dim
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[List[int]] = []
        self.input_count = 0

    def initialize_nodes(self, data: np.ndarray) -> bool:
        if len(data) < 2:
            logging.warning("GNG: Tidak cukup data untuk node awal.")
            return False
        if data.shape[1] != self.input_dim:
            logging.error(f"GNG: Dimensi data ({data.shape[1]}) tidak cocok input_dim ({self.input_dim})")
            return False
        idx = np.random.choice(len(data), 2, replace=False)
        self.nodes = [
            {'w': data[idx[0]].astype(float), 'error': 0.0, 'edges': [], 'win_count': 0, 'loss_count': 0, 'age_of_node': 0},
            {'w': data[idx[1]].astype(float), 'error': 0.0, 'edges': [], 'win_count': 0, 'loss_count': 0, 'age_of_node': 0}
        ]
        self.edges = []
        return True

    def fit(self, data: np.ndarray, num_iterations: int = 5) -> bool:
        if not self.nodes:
            if not self.initialize_nodes(data):
                logging.warning("GNG: Gagal inisialisasi node.")
                return False
        alpha_bmu = 0.5
        alpha_neighbor = 0.01
        age_increment = 1
        max_edge_age = 50
        error_decay_rate = 0.95
        for iteration in range(num_iterations):
            np.random.shuffle(data)
            for x_input in data:
                self.input_count += 1
                if not self.nodes:
                    break
                distances = np.array([np.linalg.norm(x_input - node['w']) for node in self.nodes])
                s1_idx = np.argmin(distances)
                s1 = self.nodes[s1_idx]
                s1['error'] += np.linalg.norm(x_input - s1['w'])
                s1['w'] += alpha_bmu * (x_input - s1['w'])
                s1['age_of_node'] += 1
                # Update neighbors
                for neighbor_idx in s1['edges']:
                    s_n = self.nodes[neighbor_idx]
                    s_n['w'] += alpha_neighbor * (x_input - s_n['w'])
                # Increment edge ages
                for i in range(len(self.edges)):
                    edge = self.edges[i]
                    if (edge[0] == s1_idx and edge[1] in s1['edges']) or \
                       (edge[1] == s1_idx and edge[0] in s1['edges']):
                        self.edges[i][2] += age_increment
                # Remove old edges
                new_edges = []
                for edge in self.edges:
                    if len(edge) < 3 or edge[2] <= max_edge_age:
                        new_edges.append(edge)
                self.edges = new_edges
                # Connect s1 with s2 (second closest)
                if len(self.nodes) > 1:
                    s2_idx = np.argsort(distances)[1]
                    edge_exists = False
                    for edge in self.edges:
                        if (edge[0] == s1_idx and edge[1] == s2_idx) or (edge[1] == s1_idx and edge[0] == s2_idx):
                            edge_exists = True
                            if len(edge) > 2:
                                edge[2] = 0
                            break
                    if not edge_exists:
                        self.edges.append([s1_idx, s2_idx, 0])
                        if s2_idx not in self.nodes[s1_idx]['edges']:
                            self.nodes[s1_idx]['edges'].append(s2_idx)
                        if s1_idx not in self.nodes[s2_idx]['edges']:
                            self.nodes[s2_idx]['edges'].append(s1_idx)
                # Remove isolated nodes
                nodes_to_remove: List[int] = []
                for i, node in enumerate(self.nodes):
                    if not node['edges'] and node['age_of_node'] > 10:
                        nodes_to_remove.append(i)
                for idx_to_remove in sorted(nodes_to_remove, reverse=True):
                    del self.nodes[idx_to_remove]
                    for edge in self.edges:
                        if edge[0] > idx_to_remove:
                            edge[0] -= 1
                        if edge[1] > idx_to_remove:
                            edge[1] -= 1
                    for node in self.nodes:
                        node['edges'] = [e_idx - 1 if e_idx > idx_to_remove else e_idx for e_idx in node['edges']]
                # Decay errors
                for node in self.nodes:
                    node['error'] *= error_decay_rate
        return True

# =============== FITUR & NORMALISASI ===============

def _normalize_features(features: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    normalized_features = np.zeros_like(features, dtype=float)
    for i in range(len(features)):
        val_range = max_vals[i] - min_vals[i]
        if val_range != 0:
            normalized_features[i] = (features[i] - min_vals[i]) / val_range
        else:
            normalized_features[i] = 0.5
    return normalized_features

def prepare_features_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, str, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, float]]:
    structure, swing_points = detect_structure(df)
    ob_list = detect_order_blocks_multi(df, structure_filter=structure)
    fvg_list = detect_fvg_multi(df)
    patterns = detect_engulfing(df) + detect_pinbar(df)
    boundary = get_daily_high_low(df)
    pivot = get_pivot_points(df)
    features = extract_features_full(df, structure, ob_list, fvg_list, patterns, boundary, pivot)
    return features, structure, ob_list, fvg_list, patterns, boundary, pivot

def get_gng_input_features(
    df: pd.DataFrame,
    order_blocks: List[Dict[str, Any]],
    fvg_zones: List[Dict[str, Any]],
    tf_name: str,
    gng_feature_stats: Dict[str, Dict[str, Optional[np.ndarray]]]
) -> np.ndarray:
    # Deprecated, kept for backward compatibility.
    # Gunakan get_gng_input_features_full() untuk pipeline baru.
    structure, swing_points = detect_structure(df)
    patterns = detect_engulfing(df) + detect_pinbar(df)
    boundary = get_daily_high_low(df)
    pivot = get_pivot_points(df)
    features = extract_features_full(df, structure, order_blocks, fvg_zones, patterns, boundary, pivot)
    if tf_name in gng_feature_stats and gng_feature_stats[tf_name]['min'] is not None:
        min_vals = gng_feature_stats[tf_name]['min']
        max_vals = gng_feature_stats[tf_name]['max']
        if len(features) == len(min_vals) and len(features) == len(max_vals):
            normalized_features = _normalize_features(features, min_vals, max_vals)
        else:
            logging.error(f"Fitur/Stat tidak cocok. Fitur: {len(features)}, Stat: {len(min_vals)}")
            normalized_features = features
    else:
        normalized_features = features
    return normalized_features

def get_gng_input_features_full(
    df: pd.DataFrame,
    gng_feature_stats: Dict[str, Dict[str, Optional[np.ndarray]]],
    tf_name: str
) -> np.ndarray:
    features, *_ = prepare_features_from_df(df)
    if tf_name in gng_feature_stats and gng_feature_stats[tf_name]['min'] is not None:
        min_vals = gng_feature_stats[tf_name]['min']
        max_vals = gng_feature_stats[tf_name]['max']
        if len(features) == len(min_vals) and len(features) == len(max_vals):
            normalized_features = _normalize_features(features, min_vals, max_vals)
        else:
            logging.error(f"Fitur/Stat tidak cocok. Fitur: {len(features)}, Stat: {len(min_vals)}")
            normalized_features = features
    else:
        normalized_features = features
    return normalized_features

# ================= ZONA GNG CONTEXT =================

def get_gng_context(gng_input_features: np.ndarray, gng_model: GrowingNeuralGas) -> Tuple[int, str]:
    if not gng_model or not hasattr(gng_model, 'nodes') or not gng_model.nodes:
        return 0, "No GNG Model"
    nodes_w = np.array([node['w'] for node in gng_model.nodes])
    if len(nodes_w) == 0:
        return 0, "No GNG Node"
    if gng_input_features.shape[0] != gng_model.input_dim:
        return 0, "Dimension Mismatch"
    distances = np.array([np.linalg.norm(gng_input_features - node_w) for node_w in nodes_w])
    nearest_dist = np.min(distances)
    nearest_node_idx = np.argmin(distances)
    distance_threshold_gng = 0.15
    if nearest_dist < distance_threshold_gng:
        return 1, f"Dekat zona GNG (node#{nearest_node_idx}, dist:{nearest_dist:.3f})"
    return 0, "Jauh dari zona GNG"

# ============= SAVE / LOAD MODEL & STATS =============

def save_gng_model(tf: str, model: GrowingNeuralGas, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"gng_{tf}.pkl")
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model GNG {tf} disimpan ke {path}")
    except Exception as e:
        logging.error(f"Gagal simpan model GNG {tf}: {e}")

def load_gng_model(tf: str, model_dir: str) -> Optional[GrowingNeuralGas]:
    path = os.path.join(model_dir, f"gng_{tf}.pkl")
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            logging.info(f"Model GNG {tf} dimuat dari {path}")
            return model
        except Exception as e:
            logging.warning(f"Gagal muat model GNG {tf} dari {path}: {e}")
    return None

def _calculate_feature_stats(df_list: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    features_list = []
    for df in df_list:
        feats, *_ = prepare_features_from_df(df)
        features_list.append(feats)
    features_array = np.array(features_list)
    min_vals = np.min(features_array, axis=0)
    max_vals = np.max(features_array, axis=0)
    return min_vals, max_vals

def initialize_gng_models(
    symbol: str,
    timeframes: List[str],
    model_dir: str,
    mt5_path: str,
    get_data_func,
) -> Tuple[Dict[str, GrowingNeuralGas], Dict[str, Dict[str, Optional[np.ndarray]]]]:
    gng_models: Dict[str, GrowingNeuralGas] = {}
    gng_feature_stats: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    os.makedirs(model_dir, exist_ok=True)
    # Tentukan dimensi input dari feature extractor refactor
    sample_df_for_dim = get_data_func(symbol, timeframes[0], 100, mt5_path)
    if sample_df_for_dim is None or len(sample_df_for_dim) < 20:
        logging.critical("Tidak cukup data awal untuk fitur GNG.")
        return {}, {}
    sample_features, *_ = prepare_features_from_df(sample_df_for_dim)
    input_dim = len(sample_features)
    for tf in timeframes:
        loaded_model = load_gng_model(tf, model_dir)
        stats_path = os.path.join(model_dir, f"gng_{tf}_stats.pkl")
        loaded_stats: Optional[Dict[str, np.ndarray]] = None
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'rb') as f:
                    loaded_stats = pickle.load(f)
                logging.info(f"Statistik fitur GNG {tf} dimuat dari {stats_path}")
            except Exception as e:
                logging.warning(f"Gagal muat statistik GNG {tf} dari {stats_path}: {e}")
        rebuild_model = False
        if loaded_model is None or not hasattr(loaded_model, 'input_dim') or loaded_model.input_dim != input_dim:
            rebuild_model = True
        if loaded_stats is None or loaded_stats.get('min') is None or (loaded_stats.get('min') is not None and loaded_stats['min'].shape[0] != input_dim):
            rebuild_model = True
        if rebuild_model:
            logging.info(f"Build ulang model GNG/statistik fitur untuk TF {tf}.")
            df_hist = get_data_func(symbol, tf, 1500, mt5_path)
            if df_hist is None or len(df_hist) < 100:
                gng_models[tf] = GrowingNeuralGas(max_nodes=100, input_dim=input_dim)
                gng_feature_stats[tf] = {'min': None, 'max': None}
                continue
            # Split rolling window batch untuk fitur statistik
            window = 50
            df_batches = [df_hist.iloc[i-window:i+1] for i in range(window, len(df_hist)-1)]
            min_vals, max_vals = _calculate_feature_stats(df_batches)
            gng_feature_stats[tf] = {'min': min_vals, 'max': max_vals}
            try:
                with open(stats_path, 'wb') as f:
                    pickle.dump(gng_feature_stats[tf], f)
            except Exception as e:
                logging.error(f"Gagal simpan statistik GNG {tf}: {e}")
            # Build data untuk training
            gng_data_for_fit: List[np.ndarray] = []
            for df_sub in df_batches:
                feats, *_ = prepare_features_from_df(df_sub)
                normalized_feats = _normalize_features(feats, min_vals, max_vals)
                if normalized_feats is not None and len(normalized_feats) == input_dim:
                    gng_data_for_fit.append(normalized_feats)
            model = GrowingNeuralGas(max_nodes=100, input_dim=input_dim)
            if len(gng_data_for_fit) > 1:
                model.fit(np.array(gng_data_for_fit))
                save_gng_model(tf, model, model_dir)
            gng_models[tf] = model
        else:
            gng_models[tf] = loaded_model
            gng_feature_stats[tf] = loaded_stats
    return gng_models, gng_feature_stats

# ===================== END OF MODULE =====================
