import mne
import numpy as np
from preprocessing import compute_eog_regression_coefficients


def organize_session_by_blocks(filepath, verbose=False):
    """
    Organizes a session GDF file into EOG calibration blocks and MI runs.

    Parameters
    ----------
    filepath : str
        Path to GDF file.
    verbose : bool
        Print extraction info.

    Returns
    -------
    dict
        Dictionary with keys 'eog_eyes_open', 'eog_eyes_closed', 'eog_eye_movements', 'run_0', ..., 'run_N'.
    """
    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    blocks = {}
    sfreq = raw.info['sfreq']

    # Event codes from BNCI2014_001 dataset
    EOG_EYES_OPEN = 276
    EOG_EYES_CLOSED = 277
    EOG_EYE_MOVEMENTS = 1072
    RUN_START = 32766

    def extract_segment(start_sample, end_sample):
        """Extracts a time segment from raw data."""
        tmin = start_sample / sfreq
        tmax = min(end_sample / sfreq, raw.times[-1])
        return raw.copy().crop(tmin=tmin, tmax=tmax)

    def find_event_code(target_code):
        """Finds MNE event code matching target code number."""
        for name, code in event_id.items():
            if str(target_code) in name:
                return code
        return None

    # Extract EOG calibration blocks
    eog_codes = {
        'eog_eyes_open': find_event_code(EOG_EYES_OPEN),
        'eog_eyes_closed': find_event_code(EOG_EYES_CLOSED),
        'eog_eye_movements': find_event_code(EOG_EYE_MOVEMENTS)
    }

    for block_name, code in eog_codes.items():
        if code is not None:
            mask = events[:, 2] == code
            if np.any(mask):
                event_idx = np.where(mask)[0][0]
                start_sample = events[event_idx, 0]
                end_sample = events[event_idx + 1, 0] if event_idx + 1 < len(events) else len(raw.times)
                blocks[block_name] = extract_segment(start_sample, end_sample)

    # Extract MI runs
    run_code = find_event_code(RUN_START)
    if run_code is not None:
        run_starts = events[events[:, 2] == run_code, 0]

        for i, start_sample in enumerate(run_starts):
            end_sample = run_starts[i + 1] if i + 1 < len(run_starts) else len(raw.times)
            blocks[f'run_{i}'] = extract_segment(start_sample, end_sample)

    if verbose:
        print(f"Extracted blocks: {list(blocks.keys())}")

    return blocks

def apply_eog_regression(raw, coefficients, eeg_channels, eog_channels):
    """
    Removes EOG artifacts from a Raw object using regression coefficients.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    coefficients : ndarray, shape (n_eeg_channels, n_eog_channels)
        Regression coefficients from EOG calibration.
    eeg_channels : list of str
        EEG channel names.
    eog_channels : list of str
        EOG channel names.

    Returns
    -------
    mne.io.Raw
        Cleaned Raw object.
    """
    raw_clean = raw.copy()

    eeg_data = raw_clean.get_data(picks=eeg_channels)  # (22, n_samples)
    eog_data = raw_clean.get_data(picks=eog_channels)  # (3, n_samples)

    # Clean: EEG_clean = EEG - beta @ EOG
    eeg_clean = eeg_data - coefficients @ eog_data  # (22,3) @ (3,n) = (22,n)

    # Update raw data
    for i, ch in enumerate(eeg_channels):
        idx = raw_clean.ch_names.index(ch)
        raw_clean._data[idx, :] = eeg_clean[i, :]

    return raw_clean


def preprocess_blocks(blocks, eeg_channels, eog_channels, l_freq=8, h_freq=30):
    """
    Applies EOG regression and bandpass filter to all runs.

    Parameters
    ----------
    blocks : dict
        Dictionary of Raw objects from organize_session_by_blocks.
    eeg_channels : list of str
        EEG channel names.
    eog_channels : list of str
        EOG channel names.
    l_freq : float
        Lowcut frequency in Hz.
    h_freq : float
        Highcut frequency in Hz.

    Returns
    -------
    clean_runs : dict
        Dictionary of preprocessed runs.
    coefficients : ndarray
        EOG regression coefficients.
    """
    # Compute coefficients from calibration blocks
    coefficients = compute_eog_regression_coefficients(blocks, eeg_channels, eog_channels)

    clean_runs = {}
    for key, raw in blocks.items():
        if key.startswith('run_'):
            raw_clean = apply_eog_regression(raw, coefficients, eeg_channels, eog_channels)
            raw_clean.filter(l_freq, h_freq, verbose=False)
            clean_runs[key] = raw_clean

    return clean_runs, coefficients


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

    for run_name, raw in clean_runs.items():
        events, event_id_raw = mne.events_from_annotations(raw, verbose=False)

        # Filter event_id to keep only MI cues (original codes 769-772)
        event_id = {
            name: code for name, code in event_id_raw.items()
            if name in ['769', '770', '771', '772']
        }

        # Rename to meaningful labels
        event_id_renamed = {}
        for name, code in event_id.items():
            if name == '769':
                event_id_renamed['left_hand'] = code
            elif name == '770':
                event_id_renamed['right_hand'] = code
            elif name == '771':
                event_id_renamed['feet'] = code
            elif name == '772':
                event_id_renamed['tongue'] = code

        if len(event_id_renamed) > 0:
            # Create epochs: tmin/tmax are relative to cue onset
            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id_renamed,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                preload=True,
                verbose=False
            )
            all_epochs.append(epochs)

    # Concatenate all epochs from different runs
    if len(all_epochs) > 0:
        epochs_combined = mne.concatenate_epochs(all_epochs)

        if verbose:
            print(f"Total epochs extracted: {len(epochs_combined)}")

        return epochs_combined
    else:
        return None


def filter_binary_classes(epochs, reject_artifacts=True, verbose=False):
    """
    Filters epochs to keep only left_hand and right_hand classes.

    Parameters
    ----------
    epochs : mne.Epochs or None
        Epochs containing all MI classes.
    reject_artifacts : bool
        Exclude trials marked as artifacts.
    verbose : bool
        Print class distribution info.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times) or None
        Epoch data array.
    y : ndarray, shape (n_trials,) or None
        Binary labels (0=left_hand, 1=right_hand).
    """
    if epochs is None:
        return None, None

    # Select only left and right hand trials
    epochs_binary = epochs['left_hand', 'right_hand']

    # Exclude artifact trials if requested
    if reject_artifacts and hasattr(epochs_binary, 'drop_bad'):
        epochs_binary.drop_bad()

    # Get data and labels
    X = epochs_binary.get_data()  # (n_trials, n_channels, n_times)
    y = epochs_binary.events[:, 2]  # Event codes (remapped by MNE)

    # Get actual event codes from epochs
    left_code = epochs_binary.event_id['left_hand']
    right_code = epochs_binary.event_id['right_hand']

    # Convert event codes to binary labels (0=left, 1=right)
    y_binary = (y == right_code).astype(int)

    if verbose:
        n_left = np.sum(y_binary == 0)
        n_right = np.sum(y_binary == 1)
        print(f"Binary classification: {n_left} left hand, {n_right} right hand trials")
        print(f"Data shape: {X.shape}")

    return X, y_binary


def load_session_binary_mi(filepath, eeg_channels, eog_channels,
                           tmin=0.5, tmax=2.5, l_freq=8, h_freq=30,
                           reject_artifacts=True, verbose=False):
    """
    Complete pipeline to load a session and extract binary MI data.

    Parameters
    ----------
    filepath : str
        Path to GDF file.
    eeg_channels : list of str
        EEG channel names.
    eog_channels : list of str
        EOG channel names.
    tmin : float
        Epoch start time relative to cue onset in seconds.
    tmax : float
        Epoch end time relative to cue onset in seconds.
    l_freq : float
        Bandpass lowcut frequency in Hz.
    h_freq : float
        Bandpass highcut frequency in Hz.
    reject_artifacts : bool
        Exclude artifact trials.
    verbose : bool
        Print pipeline progress.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Epoch data for binary classification.
    y : ndarray, shape (n_trials,)
        Binary labels (0=left_hand, 1=right_hand).
    epochs : mne.Epochs
        Full epochs object for further analysis.
    """
    if verbose:
        print(f"Loading session: {filepath}")

    # 1. Organize into blocks
    blocks = organize_session_by_blocks(filepath, verbose=verbose)

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

    Parameters
    ----------
    epochs : mne.Epochs or None
        Epochs containing all MI classes.
    reject_artifacts : bool
        Exclude trials marked as artifacts.
    verbose : bool
        Print class distribution info.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times) or None
        Epoch data array.
    y : ndarray, shape (n_trials,) or None
        Multiclass labels (0=left_hand, 1=right_hand, 2=feet, 3=tongue).
    """
    if epochs is None:
        return None, None

    # Exclude artifact trials if requested
    if reject_artifacts and hasattr(epochs, 'drop_bad'):
        epochs.drop_bad()

    # Get data and labels
    X = epochs.get_data()  # (n_trials, n_channels, n_times)
    y = epochs.events[:, 2]  # Event codes (remapped by MNE)

    # Get actual event codes from epochs
    left_code = epochs.event_id['left_hand']
    right_code = epochs.event_id['right_hand']
    feet_code = epochs.event_id['feet']
    tongue_code = epochs.event_id['tongue']

    # Map event codes to class labels (0-3)
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
                               reject_artifacts=True, verbose=False):
    """
    Complete pipeline to load a session and extract 4-class MI data.

    Parameters
    ----------
    filepath : str
        Path to GDF file.
    eeg_channels : list of str
        EEG channel names.
    eog_channels : list of str
        EOG channel names.
    tmin : float
        Epoch start time relative to cue onset in seconds.
    tmax : float
        Epoch end time relative to cue onset in seconds.
    l_freq : float
        Bandpass lowcut frequency in Hz.
    h_freq : float
        Bandpass highcut frequency in Hz.
    reject_artifacts : bool
        Exclude artifact trials.
    verbose : bool
        Print pipeline progress.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Epoch data for 4-class classification.
    y : ndarray, shape (n_trials,)
        Multiclass labels (0=left_hand, 1=right_hand, 2=feet, 3=tongue).
    epochs : mne.Epochs
        Full epochs object for further analysis.
    """
    if verbose:
        print(f"Loading session: {filepath}")

    # 1. Organize into blocks
    blocks = organize_session_by_blocks(filepath, verbose=verbose)

    # 2. Preprocess (EOG regression + bandpass filter)
    clean_runs, _ = preprocess_blocks(
        blocks, eeg_channels, eog_channels, l_freq, h_freq
    )

    if verbose:
        print(f"Preprocessed {len(clean_runs)} runs")

    # 3. Extract epochs with temporal window
    epochs = extract_epochs_from_runs(clean_runs, tmin, tmax, verbose=verbose)

    # 4. Get all four classes
    X, y = filter_multiclass(epochs, reject_artifacts, verbose=verbose)

    return X, y, epochs