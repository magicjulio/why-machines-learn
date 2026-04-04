import soundfile as sf
import numpy as np
import os
from pathlib import Path

print(os.getcwd())

# Resolve paths from this file's directory (experiments/)
BASE_DIR = Path(__file__).resolve().parent
babbled_path = BASE_DIR / "../static" / "5dB"
clean_path = BASE_DIR / "../static" / "clean"

def get_file(n, noise=True):
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



xn, fs_x = sf.read(get_file(1, noise=True))               # babbled version
dn, fs_d = sf.read(get_file(1, noise=False))  # desired clean version


print(xn)