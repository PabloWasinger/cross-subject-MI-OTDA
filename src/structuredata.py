import mne
import numpy as np


def organize_session_by_blocks(filepath, verbose=False):
    """Organizes a session GDF file into EOG calibration blocks and MI runs."""
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
    """Removes EOG artifacts from a Raw object using regression coefficients."""
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
    """Applies EOG regression and bandpass filter to all runs."""
    # Compute coefficients from calibration blocks
    coefficients = compute_eog_regression_coefficients(blocks, eeg_channels, eog_channels)

    clean_runs = {}
    for key, raw in blocks.items():
        if key.startswith('run_'):
            raw_clean = apply_eog_regression(raw, coefficients, eeg_channels, eog_channels)
            raw_clean.filter(l_freq, h_freq, verbose=False)
            clean_runs[key] = raw_clean

    return clean_runs, coefficients