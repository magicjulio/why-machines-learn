import sounddevice as sd
import numpy as np

def play_signal(x, fs=8000, block=True, normalize=False):
    """
    Play an audio signal array.

    Args:
        x: numpy array, shape (N,) mono or (N, C) multi-channel
        fs: sample rate (Hz), e.g. 16000
        block: if True, wait until playback finishes
        normalize: if True, scale signal to avoid clipping
    """
    x = np.asarray(x, dtype=np.float32)

    if normalize:
        peak = np.max(np.abs(x))
        if peak > 0:
            x = x / peak

    sd.play(x, samplerate=int(fs))
    if block:
        sd.wait()

def get_file(n, noise=True, babbled_path=None, clean_path=None):
    if n < 1 or n > 27:
        raise ValueError("Invalid file index. Use 1..27.")
    if noise:
        filename = f"sp{n:02d}_babble_sn5.wav"
        path = babbled_path / filename
    else:
        filename = f"sp{n:02d}.wav"
        path = clean_path / filename
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return str(path)
