import mne
import numpy as np


def compute_eog_regression_coefficients(blocks, eeg_channels, eog_channels):
    """Estimates coefficients using only blocks with ocular artifacts."""

    calibration_raws = []
    if 'eog_eyes_open' in blocks:
        calibration_raws.append(blocks['eog_eyes_open'].copy())
    if 'eog_eye_movements' in blocks:
        calibration_raws.append(blocks['eog_eye_movements'].copy())

    eog_calibration = mne.concatenate_raws(calibration_raws)

    # Extract data
    eeg_data = eog_calibration.get_data(picks=eeg_channels)  # (22, n_samples)
    eog_data = eog_calibration.get_data(picks=eog_channels)  # (3, n_samples)

    # Regression: EEG = beta @ EOG + residual
    coefficients = eeg_data @ eog_data.T @ np.linalg.inv(eog_data @ eog_data.T)


    return coefficients  

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
