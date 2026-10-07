from pathlib import Path

import librosa

def load_wavs(data_dir, sr=16000):
    train_data = {}
    test_data = {}

    for path in sorted(Path(data_dir).glob("*.wav")):
        # filename: label_speaker_index.wav
        digit, speaker, index = path.stem.split("_")

        y, _ = librosa.load(path, sr=sr, mono=True)

        if int(index) < 5:
            test_data[path.stem] = y
        else:
            train_data[path.stem] = y

    return train_data, test_data

