"""
Pipeline for comparing CSP+LDA vs EEGNet approaches in cross-subject transfer learning.

Methods compared (samplewise):
1. CSP + LDA (baseline - SC)
2. CSP + LDA + BOTDA-GL
3. EEGNet (trained on source, no adaptation)
4. EEGNet + Per-trial Fine-tuning (fair comparison - retrains every trial like BOTDA-GL)
5. EEGNet + BOTDA-GL (features from EEGNet transported with BOTDA-GL)

Follows the samplewise methodology from Peterson et al. where we incrementally
add each test trial to the transportation set BEFORE prediction.

FIXES from original version:
- FIX 1: Update validation set BEFORE prediction (critical for BOTDA-GL)
- FIX 2: Continual fine-tuning for EEGNet (fair comparison with BOTDA-GL)
- FIX 3: Stratified, reproducible train/val split for EEGNet training
- FIX 4: Proper handling of fine-tuning without val_loader
- FIX 5: Added balanced accuracy and kappa metrics
- FIX 6: Added feature dimension verification
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import timeit
import torch
import copy

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'models'))

# Existing imports from the codebase
from structuredata import load_session_binary_mi
from validation import cv_grouplasso_backward
from transfer_learning_methods import SC, Backward_GroupLasso_Transport
from subject_selection_methods import select_best_source_subject, load_subject_data

# Training imports
from training import (
    create_dataloaders, train_model, finetune_model, 
    extract_features, EEGDataset
)
from eegnet import EEGNet, BNCI2014_001_CONFIG

# Scientific imports
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
import ot

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("data")
GDF_DIR = DATA_DIR / "gdf"
LABELS_DIR = DATA_DIR / "labels"
RESULTS_DIR = Path("results") / "comparison"
MODELS_DIR = Path("models") / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [1, 2, 3, 5, 6, 7, 8, 9]  # Excluding subject 4

# Calibration trials from target
N_CALIB = 24

# CV Parameters (same as in cross-subject_samplewise.py)
CV_PARAMS = {
    'reg_e_grid': [0.1, 0.5, 1, 2, 5, 10, 20],
    'reg_cl_grid': [0.1, 0.5, 1, 2, 5, 10, 20],
    'metric': 'sqeuclidean',
    'outerkfold': 10,
    'innerkfold': None,
    'M': N_CALIB,
    'norm': None
}

# EEGNet configuration for SC and Fine-tuning (NO projection layer)
# NOTE: Reduced dropout for small datasets (144 trials is very few for deep learning)
EEGNET_CONFIG = {
    "nb_classes": 2,
    "Chans": 22,
    "Samples": 500,  # 2 seconds at 250Hz
    "dropoutRate": 0.2,  # Reduced from 0.5 - less regularization for small data
    "kernLength": 125,
    "F1": 8,
    "D": 2,
    "F2": 16,
    "feature_dim": None  # No projection - full features for classification
}

# EEGNet configuration for BOTDA-GL (WITH projection to match CSP dimensions)
EEGNET_BOTDA_CONFIG = {
    "nb_classes": 2,
    "Chans": 22,
    "Samples": 500,  # 2 seconds at 250Hz
    "dropoutRate": 0.2,  # Reduced from 0.5
    "kernLength": 125,
    "F1": 8,
    "D": 2,
    "F2": 16,
    "feature_dim": 32  # Match CSP n_components for BOTDA-GL transport
}

# Training parameters
# NOTE: Increased patience for small noisy datasets - EEG training is unstable
TRAIN_PARAMS = {
    'epochs': 500,
    'lr': 5e-4,       # Reduced LR for more stable training
    'batch_size': 16,  # Smaller batch for small dataset
    'patience': 50     # Much more patience - EEG is noisy!
}

FINETUNE_PARAMS = {
    'epochs': 10,        # Reduced for continual learning
    'lr': 1e-4,
    'batch_size': 8,
    'patience': 5
}

# Continual fine-tuning parameters (per-trial, like BOTDA-GL)
CONTINUAL_FT_EPOCHS = 5     # Quick adaptation epochs per trial

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def extract_eegnet_features_batch(model, X, device=None):
    """Extract features from EEGNet for a batch of trials."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Add channel dimension: (N, Chans, Samples) -> (N, 1, Chans, Samples)
    X_tensor = torch.from_numpy(X).float().unsqueeze(1).to(device)
    
    with torch.no_grad():
        features = model.extract_features(X_tensor)
    
    return features.cpu().numpy()


def classify_features_with_eegnet(model, features, device=None):
    """
    Classify pre-extracted features using EEGNet's classifier layer.
    
    This is used after BOTDA transport to maintain end-to-end consistency.
    
    Parameters
    ----------
    model : EEGNet
        EEGNet model with trained classifier.
    features : ndarray
        Features array of shape (N, feature_dim).
    device : torch.device, optional
        Device to use.
    
    Returns
    -------
    predictions : ndarray
        Predicted class labels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    features_tensor = torch.from_numpy(features).float().to(device)
    
    with torch.no_grad():
        logits = model.classifier(features_tensor)
        predictions = logits.argmax(dim=1).cpu().numpy()
    
    return predictions


class EEGNetClassifierWrapper:
    """
    Sklearn-compatible wrapper for EEGNet's classifier layer.
    
    Used for CV hyperparameter search with BOTDA-GL.
    The classifier is already trained - this wrapper just provides
    sklearn's fit/predict interface.
    """
    
    def __init__(self, eegnet_model, device=None):
        """
        Parameters
        ----------
        eegnet_model : EEGNet
            Pre-trained EEGNet model with classifier layer.
        device : torch.device, optional
            Device to use for inference.
        """
        self.model = eegnet_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def fit(self, X, y):
        """No-op - classifier is already trained."""
        # The classifier is pre-trained, we don't retrain it
        return self
    
    def predict(self, X):
        """Classify features using EEGNet's classifier layer."""
        features_tensor = torch.from_numpy(X).float().to(self.device)
        
        with torch.no_grad():
            logits = self.model.classifier(features_tensor)
            predictions = logits.argmax(dim=1).cpu().numpy()
        
        return predictions
    
    def __sklearn_clone__(self):
        """Return a copy for sklearn's clone function."""
        return EEGNetClassifierWrapper(self.model, self.device)


def train_eegnet_on_source(X_source, y_source, config=None, save_path=None, 
                           verbose=True, random_state=RANDOM_SEED):
    """
    Train EEGNet on source data with stratified, reproducible split.
    
    Parameters
    ----------
    config : dict, optional
        EEGNet configuration. Defaults to EEGNET_CONFIG.
    
    FIX 3: Uses stratified train/val split with fixed random state.
    """
    if config is None:
        config = EEGNET_CONFIG
    
    set_seed(random_state)
    
    model = EEGNet(**config)
    
    # FIX 3: Stratified, reproducible split
    X_train, X_val, y_train, y_val = train_test_split(
        X_source, y_source,
        test_size=0.2,
        stratify=y_source,
        random_state=random_state
    )
    
    if verbose:
        print(f"  Training split: {len(y_train)} train, {len(y_val)} val")
        print(f"  Class balance - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}")
    
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=TRAIN_PARAMS['batch_size']
    )
    
    model, history = train_model(
        model, train_loader, val_loader,
        epochs=TRAIN_PARAMS['epochs'],
        lr=TRAIN_PARAMS['lr'],
        patience=TRAIN_PARAMS['patience'],
        save_path=save_path,
        verbose=verbose
    )
    
    # Verify feature dimension
    test_features = extract_eegnet_features_batch(model, X_source[:1])
    expected_dim = config.get('feature_dim') or model._flat_features
    assert test_features.shape[1] == expected_dim, \
        f"Feature dimension mismatch: expected {expected_dim}, got {test_features.shape[1]}"
    
    if verbose:
        print(f"  Feature dimension verified: {test_features.shape[1]}")
    
    return model, history


def quick_finetune(model, X_train, y_train, epochs=CONTINUAL_FT_EPOCHS, lr=1e-4, 
                   device=None, verbose=False):
    """
    Quick fine-tuning for continual learning (no validation set).
    
    FIX 4: Proper handling without val_loader - uses fixed small number of epochs.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.train()
    
    # Create simple dataloader
    batch_size = min(8, len(y_train))
    train_loader, _ = create_dataloaders(X_train, y_train, batch_size=batch_size)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Apply max_norm constraint
            if hasattr(model, '_apply_max_norm'):
                model._apply_max_norm()
    
    model.eval()
    return model


# =============================================================================
# SAMPLEWISE EVALUATION
# =============================================================================

def evaluate_methods_samplewise(X_source, y_source, X_target, y_target, 
                                eegnet_model, eegnet_botda, cv_params, 
                                n_calib=24, verbose=True):
    """
    Evaluate all methods with incremental trial-by-trial (samplewise) adaptation.
    
    Following the methodology from Peterson et al. and adaptation_testing.py.
    
    CRITICAL FIXES:
    - FIX 1: Update validation set BEFORE prediction (for BOTDA-GL oracle labels)
    - FIX 2: Continual fine-tuning for EEGNet (fair comparison)
    
    Parameters
    ----------
    X_source : ndarray
        Source domain EEG data, shape (n_trials, n_channels, n_samples).
    y_source : ndarray
        Source domain labels.
    X_target : ndarray
        Target domain EEG data.
    y_target : ndarray
        Target domain labels.
    eegnet_model : EEGNet
        Pre-trained EEGNet model on source data (NO feature_dim - for SC/FT).
    eegnet_botda : EEGNet
        Pre-trained EEGNet model with feature_dim projection (for BOTDA-GL).
    cv_params : dict
        Cross-validation parameters.
    n_calib : int
        Number of calibration trials from target.
    verbose : bool
        Print progress.
    
    Returns
    -------
    predictions : dict
        Dictionary with method names -> list of predictions.
    times : dict
        Dictionary with method names -> list of execution times.
    y_test : ndarray
        True labels for test trials.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # =========================================================================
    # STEP 1: Train CSP+LDA on source (baseline classifier)
    # =========================================================================
    csp = CSP(n_components=6, reg='empirical', log=True, norm_trace=False, cov_est='epoch')
    G_source = csp.fit_transform(X_source, y_source)
    
    clf_base = LinearDiscriminantAnalysis()
    clf_base.fit(G_source, y_source)
    
    # =========================================================================
    # STEP 2: Extract EEGNet features from source (using BOTDA model with projection)
    # =========================================================================
    # Use eegnet_botda (with feature_dim=6) for BOTDA-GL feature extraction
    F_source = extract_eegnet_features_batch(eegnet_botda, X_source, device)
    
    
    if verbose:
        print(f"Feature dimensions - CSP: {G_source.shape[1]}, EEGNet_BOTDA: {F_source.shape[1]}")
    
    # Create sklearn-compatible wrapper for EEGNet's classifier (for CV and prediction)
    # This uses the TRAINED classifier from eegnet_botda, not a separate LDA
    clf_eegnet = EEGNetClassifierWrapper(eegnet_botda, device)
    
    # =========================================================================
    # STEP 3: Calibration split
    # =========================================================================
    X_val_raw = X_target[:n_calib].copy()
    y_val = y_target[:n_calib].copy()
    X_test_raw = X_target[n_calib:]
    y_test = y_target[n_calib:]
    
    # Transform calibration to CSP space
    G_val = csp.transform(X_val_raw)
    
    # Extract EEGNet features from calibration (using BOTDA model with projection)
    F_val = extract_eegnet_features_batch(eegnet_botda, X_val_raw, device)
    
    # =========================================================================
    # STEP 4: Cross-validation for BOTDA-GL hyperparameters (CSP)
    # =========================================================================
    if verbose:
        print("Running CV for CSP+BOTDA-GL hyperparameters...")
    
    G_bg_subsample, _, reg_bg_csp = cv_grouplasso_backward(
        cv_params['reg_e_grid'], cv_params['reg_cl_grid'],
        G_source, y_source, G_val, y_val, clf_base,
        metric=cv_params['metric'],
        outerkfold=cv_params['outerkfold'],
        innerkfold=cv_params['innerkfold'],
        M=cv_params['M'],
        norm=cv_params['norm'],
        verbose=verbose
    )
    
    # =========================================================================
    # STEP 5: Cross-validation for BOTDA-GL hyperparameters (EEGNet features)
    # =========================================================================
    if verbose:
        print("Running CV for EEGNet+BOTDA-GL hyperparameters...")
    
    F_bg_subsample, _, reg_bg_eegnet = cv_grouplasso_backward(
        cv_params['reg_e_grid'], cv_params['reg_cl_grid'],
        F_source, y_source, F_val, y_val, clf_eegnet,
        metric=cv_params['metric'],
        outerkfold=cv_params['outerkfold'],
        innerkfold=cv_params['innerkfold'],
        M=cv_params['M'],
        norm=cv_params['norm'],
        verbose=verbose
    )
    
    # =========================================================================
    # STEP 6: Initialize EEGNet for fine-tuning (FIX 2: Continual learning)
    # =========================================================================
    if verbose:
        print("Initializing EEGNet for continual fine-tuning...")
    
    # Create a copy for fine-tuning that will be updated incrementally
    eegnet_finetuned = EEGNet(**EEGNET_CONFIG)
    eegnet_finetuned.load_state_dict(eegnet_model.state_dict())
    
    # Freeze temporal layers for fine-tuning stability
    eegnet_finetuned.freeze_temporal_layers()
    
    # Initial fine-tuning with calibration data
    eegnet_finetuned = quick_finetune(
        eegnet_finetuned, X_val_raw, y_val,
        epochs=FINETUNE_PARAMS['epochs'],
        lr=FINETUNE_PARAMS['lr'],
        device=device
    )
    
    if verbose:
        print(f"  Initial fine-tuning complete with {n_calib} calibration trials")
    
    # =========================================================================
    # STEP 7: Samplewise evaluation loop
    # =========================================================================
    methods = ['CSP_LDA', 'CSP_LDA_BOTDA', 'EEGNet', 'EEGNet_FT', 'EEGNet_BOTDA']
    predictions = {m: [] for m in methods}
    times = {m: [] for m in methods}
    
    if verbose:
        print(f"\nStarting samplewise evaluation ({len(y_test)} test trials)...")
    
    for trial_idx in range(1, len(y_test) + 1):
        if verbose and trial_idx % 20 == 1:
            print(f"  Processing trial {trial_idx}/{len(y_test)}")
        
        # Current test trial
        X_test_trial = X_test_raw[trial_idx-1:trial_idx]
        y_test_trial = y_test[trial_idx-1:trial_idx]
        
        # Transform to CSP space
        G_test = csp.transform(X_test_trial)
        
        # Extract EEGNet features for test trial (needed for F_val update)
        F_test = extract_eegnet_features_batch(eegnet_botda, X_test_trial, device)
        
        # =================================================================
        # FIX 1: Update validation set BEFORE prediction
        # This is critical for BOTDA-GL which uses oracle labels
        # =================================================================
        X_val_raw = np.vstack((X_val_raw, X_test_trial))
        y_val = np.hstack((y_val, y_test_trial))
        G_val = np.vstack((G_val, G_test))
        F_val = np.vstack((F_val, F_test))
        
        # -----------------------------------------------------------------
        # Method 1: CSP + LDA (baseline - no adaptation)
        # -----------------------------------------------------------------
        start = timeit.default_timer()
        pred_csp, _ = SC(G_test, clf_base)
        times['CSP_LDA'].append(timeit.default_timer() - start)
        predictions['CSP_LDA'].append(pred_csp)
        
        # -----------------------------------------------------------------
        # Method 2: CSP + LDA + BOTDA-GL (uses oracle labels via y_val)
        # -----------------------------------------------------------------
        pred_botda, time_botda = Backward_GroupLasso_Transport(
            G_bg_subsample, reg_bg_csp, G_val, y_val, G_test, clf_base, 
            cv_params['metric']
        )
        times['CSP_LDA_BOTDA'].append(time_botda)
        predictions['CSP_LDA_BOTDA'].append(pred_botda)
        
        # -----------------------------------------------------------------
        # Method 3: EEGNet (no adaptation - frozen source model)
        # -----------------------------------------------------------------
        start = timeit.default_timer()
        eegnet_model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_test_trial).float().unsqueeze(1).to(device)
            logits = eegnet_model(X_tensor)
            pred_eegnet = logits.argmax(dim=1).cpu().numpy()
        times['EEGNet'].append(timeit.default_timer() - start)
        predictions['EEGNet'].append(pred_eegnet)
        
        # -----------------------------------------------------------------
        # Method 4: EEGNet + Continual Fine-tuning (FIX 2: Fair comparison)
        # Fine-tune with the NEW trial only (true continual learning)
        # -----------------------------------------------------------------
        start = timeit.default_timer()
        
        # FIX 2: Fine-tune only with the NEW trial (not all accumulated data)
        # This is true continual learning - model adapts incrementally
        eegnet_finetuned = quick_finetune(
            eegnet_finetuned, X_test_trial, y_test_trial, # PREGUNTAR A TRINIDAD MONREAL :)
            epochs=CONTINUAL_FT_EPOCHS,
            lr=FINETUNE_PARAMS['lr'],
            device=device
        )
        
        eegnet_finetuned.eval()
        with torch.no_grad():
            logits_ft = eegnet_finetuned(X_tensor)
            pred_eegnet_ft = logits_ft.argmax(dim=1).cpu().numpy()
        times['EEGNet_FT'].append(timeit.default_timer() - start)
        predictions['EEGNet_FT'].append(pred_eegnet_ft)
        
        # -----------------------------------------------------------------
        # Method 5: EEGNet + BOTDA-GL (uses oracle labels via y_val)
        # -----------------------------------------------------------------
        start = timeit.default_timer()
        
        # BOTDA-GL transport on EEGNet features
        botda = ot.da.SinkhornL1l2Transport(
            metric=cv_params['metric'],
            reg_e=reg_bg_eegnet[0],
            reg_cl=reg_bg_eegnet[1]
        )
        botda.fit(Xs=F_val, ys=y_val, Xt=F_bg_subsample)

        # Use F_test already calculated earlier (no need to recalculate)
        F_test_transported = botda.transform(Xs=F_test)
        
        if np.isnan(F_test_transported).any():
            F_test_transported = np.nan_to_num(F_test_transported)
        
        # Use EEGNet's classifier (not LDA) for end-to-end consistency
        pred_eegnet_botda = classify_features_with_eegnet(
            eegnet_botda, F_test_transported, device
        )
        times['EEGNet_BOTDA'].append(timeit.default_timer() - start)
        predictions['EEGNet_BOTDA'].append(pred_eegnet_botda)
    
    return predictions, times, y_test


def calculate_accuracies(predictions, y_test, print_results=True):
    """
    Calculate accuracy metrics for each method.
    
    FIX 5: Added balanced accuracy and Cohen's kappa.
    """
    results = []
    
    for method, preds in predictions.items():
        preds_array = np.array(preds).flatten()
        correct = np.sum(preds_array == y_test)
        total = len(y_test)
        
        results.append({
            'Method': method,
            'Accuracy': (correct / total) * 100,
            'Balanced_Acc': balanced_accuracy_score(y_test, preds_array) * 100,
            'Kappa': cohen_kappa_score(y_test, preds_array),
            'Correct': correct,
            'Total': total
        })
    
    df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    
    if print_results:
        print("\n" + "="*70)
        print("Results:")
        print("="*70)
        print(df.to_string(index=False))
    
    return df


def save_results(source_id, target_id, predictions, times, y_test):
    """Save results for a source-target pair."""
    base_name = f"comparison_src_{source_id:02d}_tgt_{target_id:02d}"
    
    # Save accuracies
    acc_df = calculate_accuracies(predictions, y_test, print_results=True)
    acc_df.to_csv(RESULTS_DIR / f"{base_name}_accuracies.csv", index=False)
    
    # Save predictions
    pred_data = {'trial': list(range(1, len(y_test) + 1)), 'true_label': y_test}
    for method, preds in predictions.items():
        pred_data[method] = np.array(preds).flatten()
    pd.DataFrame(pred_data).to_csv(RESULTS_DIR / f"{base_name}_predictions.csv", index=False)
    
    # Save times
    times_data = {'trial': list(range(1, len(y_test) + 1))}
    for method, t in times.items():
        times_data[method] = t
    pd.DataFrame(times_data).to_csv(RESULTS_DIR / f"{base_name}_times.csv", index=False)
    
    print(f"Results saved to {RESULTS_DIR}")
    return acc_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("CROSS-SUBJECT COMPARISON: CSP+LDA vs EEGNet (SAMPLEWISE)")
    print("="*70)
    print("\nMethodology: Peterson et al. samplewise with oracle labels")
    print("- Validation set updated BEFORE prediction (includes current trial)")
    print("- BOTDA-GL and EEGNet_FT both have access to incremental oracle labels")
    print("- Fair comparison between domain adaptation and continual learning")
    print("="*70)
    
    # Set global seed
    set_seed(RANDOM_SEED)
    
    # Select best source subject
    print("\nSelecting best source subject...")
    best_source_id, ranking_df = select_best_source_subject(SUBJECTS, session='T')
    print(f"Best source: Subject {best_source_id}")
    
    # Load source data
    print(f"\nLoading source data (Subject {best_source_id})...")
    X_source, y_source = load_subject_data(best_source_id, session='T')
    print(f"Source shape: {X_source.shape}")
    print(f"Source class distribution: {np.bincount(y_source)}")
    
    # Train EEGNet models on source
    print("\n" + "-"*70)
    print("Training EEGNet models on source subject...")
    print("-"*70)
    
    # Model 1: EEGNet for SC and Fine-tuning (NO feature_dim projection)
    print("\n1. Training EEGNet for SC/FT (no projection)...")
    model_path = MODELS_DIR / f'eegnet_source_{best_source_id:02d}.pt'
    eegnet_model, history = train_eegnet_on_source(
        X_source, y_source,
        config=EEGNET_CONFIG,  # No feature_dim
        save_path=str(model_path),
        verbose=True,
        random_state=RANDOM_SEED
    )
    if history['val_acc']:
        print(f"  EEGNet (SC/FT) trained. Best val accuracy: {max(history['val_acc']):.2f}%")
    
    # Model 2: EEGNet for BOTDA-GL (WITH feature_dim=6 to match CSP)
    print("\n2. Training EEGNet for BOTDA-GL (with projection to feature_dim=6)...")
    model_botda_path = MODELS_DIR / f'eegnet_botda_source_{best_source_id:02d}.pt'
    eegnet_botda, history_botda = train_eegnet_on_source(
        X_source, y_source,
        config=EEGNET_BOTDA_CONFIG,  # With feature_dim=6
        save_path=str(model_botda_path),
        verbose=True,
        random_state=RANDOM_SEED
    )
    if history_botda['val_acc']:
        print(f"  EEGNet (BOTDA) trained. Best val accuracy: {max(history_botda['val_acc']):.2f}%")
    
    # Evaluate on all target subjects
    all_results = {}
    targets = [s for s in SUBJECTS if s != best_source_id]
    
    for target_id in targets:
        print(f"\n{'='*70}")
        print(f"Evaluating Transfer: S{best_source_id} -> S{target_id}")
        print(f"{'='*70}")
        
        try:
            X_target, y_target = load_subject_data(target_id, session='T')
            print(f"Target shape: {X_target.shape}")
            print(f"Target class distribution: {np.bincount(y_target)}")
            
            predictions, times, y_test = evaluate_methods_samplewise(
                X_source, y_source,
                X_target, y_target,
                eegnet_model,      # For SC and Fine-tuning (no projection)
                eegnet_botda,      # For BOTDA-GL (with projection to feature_dim=6)
                CV_PARAMS,
                n_calib=N_CALIB,
                verbose=True
            )
            
            acc_df = save_results(best_source_id, target_id, predictions, times, y_test)
            all_results[target_id] = acc_df
            
        except Exception as e:
            print(f"ERROR with target {target_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    summary_data = []
    for target_id, acc_df in all_results.items():
        row = {'Target': target_id}
        for _, r in acc_df.iterrows():
            row[r['Method']] = r['Accuracy']
            row[f"{r['Method']}_Kappa"] = r['Kappa']
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(RESULTS_DIR / f'summary_src_{best_source_id:02d}.csv', index=False)
    
    # Print accuracy columns only
    acc_cols = ['Target'] + [c for c in summary_df.columns if not c.endswith('_Kappa')]
    print(summary_df[acc_cols].to_string(index=False))
    
    # Mean across targets
    print("\n" + "-"*70)
    print("Mean Accuracy (%):")
    methods = [c for c in summary_df.columns if c != 'Target' and not c.endswith('_Kappa')]
    for method in methods:
        mean = summary_df[method].mean()
        std = summary_df[method].std()
        print(f"  {method:20s}: {mean:.2f} ± {std:.2f}")
    
    print("\nMean Kappa:")
    kappa_methods = [c for c in summary_df.columns if c.endswith('_Kappa')]
    for method in kappa_methods:
        mean = summary_df[method].mean()
        std = summary_df[method].std()
        print(f"  {method:25s}: {mean:.3f} ± {std:.3f}")
    
    print("\n" + "="*70)
    print(f"Results saved to {RESULTS_DIR}")
    print("="*70)
    
    return summary_df


if __name__ == "__main__":
    main()