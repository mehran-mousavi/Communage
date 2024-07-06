import io
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Vol, Resample
import json


def get_gui_dark_stylesheet():
    """
    Return dark Style sheet for app GUI
    """
    return """
            QMainWindow {
                background-color: #2b2b2b;
            }
            QPushButton, QComboBox {
                color: #ffffff;
                background-color: #333333;
                border: 1px solid #363636;
                padding: 10px;
            }
            QLabel {
                border: none;
                background-color: none;
                color: #939393;
            }
            QPushButton {
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:pressed {
                background-color: #5bc82e;
            }
            QComboBox {
                border-radius: 3px;
                padding: 5px;
                min-width: 6em;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 40px;
                border-left-width: 1px;
                border-left-color: #333333;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
                background-color: #2d2d2d;
            }
            QComboBox::down-arrow {
                image: url(resources/arrow-icon.png);
            }
            QComboBox QAbstractItemView {
                border: 2px solid darkgray;
                selection-background-color: #6a6a6a;
            }
            QComboBox:disabled {
                color: #7a7a7a;
                background-color: #2e2e2e;
                border: 1px solid #2e2e2e;
            }
            QTextEdit {
                color: #ffffff;
                border: 1px solid #2d2d2d;
                border-radius: 8px;
                padding-top: 5px;
                padding-left: 5px;
                padding-right: 5px;
                padding-bottom: 5px;
                text-align: center;
            }
        """


def get_languages():
    """
    Returns a list of languages supported by app using settings.json content.
    """
    with open("resources/settings.json", "r") as settings_file:
        settings = json.load(settings_file)
        language_names = settings.get("language_names", {})

    languages = []
    for code, name in language_names.items():
        languages.append({"name": name, "code": code})

    return languages


def int2float(sound):
    """
    Converts a NumPy array of integers to a PyTorch tensor of floats.

    Args:
        sound (np.ndarray): The NumPy array of integers.

    Returns:
        torch.Tensor: The PyTorch tensor of floats.
    """
    _sound = np.copy(sound)
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype("float32")
    if abs_max > 0:
        _sound *= 1 / abs_max
    return torch.from_numpy(_sound.squeeze())


def pre_process_audio(
    tensor: torch.Tensor,
    amplify_level: float = 2.0,
    original_sampling_rate: int = 16000,
    target_sample_rate: int = 8000,
):
    """
    Preprocesses audio data by applying volume adjustment, normalization, and resampling.

    Args:
        tensor (torch.Tensor): The audio data as a PyTorch tensor.
        amplify_level (float, optional): The amplification level. Defaults to 2.0.
        original_sampling_rate (int, optional): The original sample rate of the audio. Defaults to 16000.
        target_sample_rate (int, optional): The target sample rate for resampling. Defaults to 8000.

    Returns:
        bytes: The preprocessed audio data as a FLAC-encoded byte string.
    """
    # Apply volume adjustment (amplification)
    vol_transform = Vol(amplify_level)
    amplified_tensor = vol_transform(tensor)
    # Normalize the audio to a consistent volume level
    normalized_tensor = torchaudio.functional.gain(amplified_tensor, gain_db=0.0)
    # Resample from original_sampling_rate to target_sample_rate
    resample_transform = Resample(
        orig_freq=original_sampling_rate, new_freq=target_sample_rate
    )
    resampled_waveform = resample_transform(normalized_tensor)
    resampled_waveform = (
        resampled_waveform.unsqueeze(0)
        if resampled_waveform.dim() == 1
        else resampled_waveform
    )
    # Save the processed audio to an in-memory buffer
    buffer = io.BytesIO()
    torchaudio.save(
        buffer,
        resampled_waveform,
        target_sample_rate,
        bits_per_sample=16,
        format="flac",
    )
    # Get the byte string of the FLAC audio data
    buffer.seek(0)
    flac_data = buffer.read()
    return flac_data
