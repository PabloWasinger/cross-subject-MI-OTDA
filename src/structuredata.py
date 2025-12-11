"""
Data loading and preprocessing for BNCI2014_001 dataset.

Handles:
- GDF file loading with proper event extraction
- True label mapping for evaluation files
- EOG artifact removal via regression
- Epoch extraction for MI classification
"""

import mne
import numpy as np
import scipy.io
from pathlib import Path
from preprocessing import preprocess_blocks


def load_true_labels(mat_filepath):
    """
    Loads true labels from MAT file.

    Parameters
    ----------
    mat_filepath : str
        Path to MAT file containing true labels.

    Returns
    -------
    labels : ndarray, shape (288,)
        Class labels (1=left, 2=right, 3=feet, 4=tongue).
    """
    mat = scipy.io.loadmat(mat_filepath)
    labels = mat['classlabel'].flatten()
    return labels


def apply_true_labels_to_raw(raw, true_labels, verbose=False):
    """
    Replaces 783 (unknown cue) events with true class labels.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object with 783 events.
    true_labels : ndarray, shape (n_trials,)
        True class labels (1=left, 2=right, 3=feet, 4=tongue).
    verbose : bool
        Print replacement info.

    Returns
    -------
    mne.io.Raw
        Raw object with corrected event annotations.
    """
    raw_corrected = raw.copy()
    events, event_id = mne.events_from_annotations(raw_corrected, verbose=False)

    
    cue_unknown_code = None
    for name, code in event_id.items():
        if name == '783':  
            cue_unknown_code = code
            break

    if cue_unknown_code is None:
        if verbose:
            print("No 783 events found, assuming labels are already present")
        return raw_corrected

    
    unknown_indices = np.where(events[:, 2] == cue_unknown_code)[0]

    if len(unknown_indices) != len(true_labels):

        print(f"Warning: Mismatch: {len(unknown_indices)} unknown events but {len(true_labels)} labels provided. Truncating to min.")
        min_len = min(len(unknown_indices), len(true_labels))
        unknown_indices = unknown_indices[:min_len]
        true_labels = true_labels[:min_len]

    if verbose:
        print(f"Replacing {len(unknown_indices)} event 783 with true class labels")

    new_onsets = []
    new_durations = []
    new_descriptions = []

    label_to_desc = {1: '769', 2: '770', 3: '771', 4: '772'}

    trial_idx = 0

    unknown_indices_set = set(unknown_indices)
    
    current_unknown_count = 0
    for annot in raw_corrected.annotations:    
        current_unknown_count = 0
        desc = annot['description']
        if desc == '783':
            if current_unknown_count < len(true_labels):
                new_desc = label_to_desc[true_labels[current_unknown_count]]
                new_descriptions.append(new_desc)
                current_unknown_count += 1
            else:
                new_descriptions.append(desc) 
        else:
            new_descriptions.append(desc)

        new_onsets.append(annot['onset'])
        new_durations.append(annot['duration'])

    new_annotations = mne.Annotations(
        onset=new_onsets,
        duration=new_durations,
        description=new_descriptions,
        orig_time=raw_corrected.annotations.orig_time
    )

    raw_corrected.set_annotations(new_annotations)

    return raw_corrected


def organize_session_by_blocks(filepath, true_labels_path=None, verbose=False):
    """
    Organizes a session GDF file into EOG calibration blocks and MI runs.

    Parameters
    ----------
    filepath : str
        Path to GDF file.
    true_labels_path : str, optional
        Path to MAT file with true labels (for evaluation files).
    verbose : bool
        Print extraction info.

    Returns
    -------
    dict
        Dictionary with keys 'eog_eyes_open', 'eog_eyes_closed', 'eog_eye_movements', 'run_0', ..., 'run_5'.
    """
    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)

    if true_labels_path is not None:
        true_labels = load_true_labels(true_labels_path)
        raw = apply_true_labels_to_raw(raw, true_labels, verbose=verbose)

    events, event_id = mne.events_from_annotations(raw, verbose=False)

    blocks = {}
    sfreq = raw.info['sfreq']

    def extract_segment(start_sample, end_sample):
        """Extracts a time segment from raw data."""
        tmin = start_sample / sfreq
        tmax = min(end_sample / sfreq, raw.times[-1])
        if tmax > tmin: 
             return raw.copy().crop(tmin=tmin, tmax=tmax - 0.001)
        return raw.copy().crop(tmin=tmin, tmax=tmax)

    def find_event_code(target_code):
        """Finds MNE event code matching target code number (exact match)."""
        for name, code in event_id.items():
            if name == str(target_code): 
                return code
        return None

    # Event codes from BNCI2014_001 dataset
    EOG_EYES_OPEN = 276
    EOG_EYES_CLOSED = 277
    EOG_EYE_MOVEMENTS = 1072
    RUN_START = 32766
    TRIAL_START = 768

    # Extract EOG calibration blocks
    eog_codes = {
        'eog_eyes_open': find_event_code(EOG_EYES_OPEN),
        'eog_eyes_closed': find_event_code(EOG_EYES_CLOSED),
        'eog_eye_movements': find_event_code(EOG_EYE_MOVEMENTS)
    }

    # Find when EOG section ends (last EOG event)
    eog_end_sample = 0
    for block_name, code in eog_codes.items():
        if code is not None:
            mask = events[:, 2] == code
            if np.any(mask):
                event_idx = np.where(mask)[0][0]
                start_sample = events[event_idx, 0]
                # Find end: next event after this one
                end_sample = events[event_idx + 1, 0] if event_idx + 1 < len(events) else int(len(raw.times) * sfreq)
                blocks[block_name] = extract_segment(start_sample, end_sample)
                eog_end_sample = max(eog_end_sample, end_sample)

    # Extract MI runs - only those AFTER EOG section and containing trials
    run_code = find_event_code(RUN_START)
    trial_code = find_event_code(TRIAL_START)
    
    if run_code is not None:
        run_start_samples = events[events[:, 2] == run_code, 0]
        

        mi_run_starts = run_start_samples[run_start_samples >= eog_end_sample]
        
        if verbose:
            print(f"Found {len(run_start_samples)} total run markers, {len(mi_run_starts)} are MI runs (after EOG)")
        
        
        if trial_code is not None:
            trial_samples = events[events[:, 2] == trial_code, 0]
        else:
            trial_samples = np.array([])
        
        run_idx = 0
        for i, start_sample in enumerate(mi_run_starts):

            original_idx = np.where(run_start_samples == start_sample)[0][0]
            
            if original_idx + 1 < len(run_start_samples):
                end_sample = run_start_samples[original_idx + 1]
            else:
                end_sample = int(len(raw.times) * sfreq)
            
            trials_in_run = np.sum((trial_samples >= start_sample) & (trial_samples < end_sample))
            
            if trials_in_run > 0:  
                blocks[f'run_{run_idx}'] = extract_segment(start_sample, end_sample)
                if verbose:
                    print(f"  run_{run_idx}: {trials_in_run} trials")
                run_idx += 1

    if verbose:
        print(f"Extracted blocks: {list(blocks.keys())}")

    return blocks



def extract_epochs_from_runs(clean_runs, tmin=0.5, tmax=2.5, baseline=None, verbose=False):
    """
    Extracts epochs from clean runs based on motor imagery cues.

    Parameters
    ----------
    clean_runs : dict
        Dictionary of preprocessed Raw objects.
    tmin : float
        Start time relative to cue onset in seconds.
    tmax : float
        End time relative to cue onset in seconds.
    baseline : tuple or None
        Baseline correction period.
    verbose : bool
        Print extraction info.

    Returns
    -------
    mne.Epochs or None
        Concatenated epochs from all runs, or None if no events found.
    """
    all_epochs = []

    fixed_event_id = {
        'left_hand': 1,
        'right_hand': 2,
        'feet': 3,
        'tongue': 4
    }

    MI_EVENT_NAMES = {'769', '770', '771', '772'}

    for run_name in sorted(clean_runs.keys()):  
        raw = clean_runs[run_name]
        events, event_id_raw = mne.events_from_annotations(raw, verbose=False)

        
        event_mapping = {}
        for name, code in event_id_raw.items():
            if name == '769':
                event_mapping[code] = fixed_event_id['left_hand']
            elif name == '770':
                event_mapping[code] = fixed_event_id['right_hand']
            elif name == '771':
                event_mapping[code] = fixed_event_id['feet']
            elif name == '772':
                event_mapping[code] = fixed_event_id['tongue']

        if len(event_mapping) > 0:
            
            # 1. Identify which raw codes correspond to MI classes
            valid_raw_codes = list(event_mapping.keys())
            
            # 2. Keep ONLY those events
            mask_valid = np.isin(events[:, 2], valid_raw_codes)
            events_mi_only = events[mask_valid].copy()
            
            # 3. Now it is safe to remap codes
            for old_code, new_code in event_mapping.items():
                events_mi_only[events_mi_only[:, 2] == old_code, 2] = new_code

            if len(events_mi_only) == 0:
                continue

            if verbose:
                print(f"  {run_name}: {len(events_mi_only)} MI events")

            # Create epochs with fixed event_id
            epochs = mne.Epochs(
                raw,
                events_mi_only,
                event_id=fixed_event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                preload=True,
                verbose=False,
                event_repeated='drop'  
            )
            all_epochs.append(epochs)

    
    if len(all_epochs) > 0:
        epochs_combined = mne.concatenate_epochs(all_epochs)

        if verbose:
            print(f"Total epochs extracted: {len(epochs_combined)}")
            
            for class_name, class_code in fixed_event_id.items():
                count = np.sum(epochs_combined.events[:, 2] == class_code)
                print(f"  {class_name}: {count}")

        return epochs_combined
    else:
        return None


def filter_binary_classes(epochs, reject_artifacts=True, verbose=False):
    """
    Filters epochs to keep only left_hand and right_hand classes.
    """
    if epochs is None:
        return None, None


    epochs_binary = epochs['left_hand', 'right_hand']

    
    if reject_artifacts and hasattr(epochs_binary, 'drop_bad'):
        epochs_binary.drop_bad()

    
    eeg_picks = [ch for ch in epochs_binary.ch_names if ch.startswith('EEG')]
    X = epochs_binary.get_data(picks=eeg_picks)  
    y = epochs_binary.events[:, 2]  

    
    left_code = epochs_binary.event_id['left_hand']
    right_code = epochs_binary.event_id['right_hand']

    y_binary = (y == right_code).astype(int)

    if verbose:
        n_left = np.sum(y_binary == 0)
        n_right = np.sum(y_binary == 1)
        print(f"Binary classification: {n_left} left hand, {n_right} right hand trials")
        print(f"Data shape: {X.shape}")

    return X, y_binary


def load_session_binary_mi(filepath, eeg_channels, eog_channels,
                           tmin=0.5, tmax=2.5, l_freq=8, h_freq=30,
                           reject_artifacts=True, true_labels_path=None, verbose=False):
    """
    Complete pipeline to load a session and extract binary MI data.
    """
    if verbose:
        print(f"Loading session: {filepath}")

    # 1. Organize into blocks
    blocks = organize_session_by_blocks(filepath, true_labels_path=true_labels_path, verbose=verbose)

    # 2. Preprocess (EOG regression + bandpass filter)
    clean_runs, coefficients = preprocess_blocks(
        blocks, eeg_channels, eog_channels, l_freq, h_freq
    )

    if verbose:
        print(f"Preprocessed {len(clean_runs)} runs")

    # 3. Extract epochs with temporal window
    epochs = extract_epochs_from_runs(clean_runs, tmin, tmax, verbose=verbose)

    # 4. Filter to binary classes (left vs right)
    X, y = filter_binary_classes(epochs, reject_artifacts, verbose=verbose)

    return X, y, epochs


def filter_multiclass(epochs, reject_artifacts=True, verbose=False):
    """
    Filters epochs to keep all four MI classes.
    """
    if epochs is None:
        return None, None

    
    if reject_artifacts and hasattr(epochs, 'drop_bad'):
        epochs.drop_bad()

    eeg_picks = [ch for ch in epochs.ch_names if ch.startswith('EEG')]
    X = epochs.get_data(picks=eeg_picks)  
    y = epochs.events[:, 2]  

    left_code = epochs.event_id['left_hand']
    right_code = epochs.event_id['right_hand']
    feet_code = epochs.event_id['feet']
    tongue_code = epochs.event_id['tongue']

    y_multiclass = np.zeros(len(y), dtype=int)
    y_multiclass[y == left_code] = 0
    y_multiclass[y == right_code] = 1
    y_multiclass[y == feet_code] = 2
    y_multiclass[y == tongue_code] = 3

    if verbose:
        for cls in range(4):
            n_cls = np.sum(y_multiclass == cls)
            cls_names = ['left_hand', 'right_hand', 'feet', 'tongue']
            print(f"Class {cls} ({cls_names[cls]}): {n_cls} trials")
        print(f"Data shape: {X.shape}")

    return X, y_multiclass


def load_session_multiclass_mi(filepath, eeg_channels, eog_channels,
                               tmin=0.5, tmax=2.5, l_freq=8, h_freq=30,
                               reject_artifacts=True, true_labels_path=None, verbose=False):
    """
    Complete pipeline to load a session and extract 4-class MI data.
    """
    if verbose:
        print(f"Loading session: {filepath}")

    
    blocks = organize_session_by_blocks(filepath, true_labels_path=true_labels_path, verbose=verbose)

    clean_runs, _ = preprocess_blocks(
        blocks, eeg_channels, eog_channels, l_freq, h_freq
    )

    if verbose:
        print(f"Preprocessed {len(clean_runs)} runs")

    epochs = extract_epochs_from_runs(clean_runs, tmin, tmax, verbose=verbose)

    X, y = filter_multiclass(epochs, reject_artifacts, verbose=verbose)

    return X, y, epochs