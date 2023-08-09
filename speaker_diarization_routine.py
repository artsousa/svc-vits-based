import os
import math
import numpy as np
import torchaudio
from pyannote.audio import Pipeline

PATH = os.getcwd()

VAD_RESULTS = {
    "lula_podpah.wav": (
        [0.5, 21.6, 29.4, 52.9, 61.4, 74.0, 94.1, 98.8, 103.1, 118.1, 127.3, 128.5, 143.8, 157.6, 183.8, 188.2, 191.6,
         199.8, 202.2, 212.0, 226.0, 245.1, 253.8, 270.8, 304.5, 312.4, 313.5, 319.5, 342.2, 353.6, 383.4, 388.3, 421.8,
         423.0, 452.0, 455.0, 504.3, 515.7, 532.7, 541.9, ],
        [19.9, 51.7, 30.5, 74.0, 63.0, 100.0, 94.9, 102.0, 117.3, 126.5, 156.9, 130.3, 143.9, 182.7, 186.3, 190.3,
         200.7, 200.2, 210.9, 225.3, 243.7, 252.8, 303.2, 273.3, 311.6, 312.5, 318.5, 340.9, 352.3, 387.4, 384.6, 420.9,
         450.8, 423.0, 453.5, 514.8, 504.8, 531.8, 540.9, 543.1, ],
        ["speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_00", "speaker_SPEAKER_01",
         "speaker_SPEAKER_01", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00",
         "speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_01", "speaker_SPEAKER_00", "speaker_SPEAKER_00",
         "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_00", "speaker_SPEAKER_00",
         "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_00",
         "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00",
         "speaker_SPEAKER_01", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_00",
         "speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00"]
    ),
    "dilma_1.wav": (
        [0.5, 18.3, 21.5, 39.0, 90.2, 95.5, 100.2, 113.6, 117.0, 132.6, 140.4, 146.4, 153.9, 183.7, 207.9, 211.0, 264.7,
         298.5, 311.8, 320.0, 329.6, 348.4, 380.4, 406.7, 418.2, 424.4, 432.9, 436.2, 442.6, 453.1, 459.6, 481.6, 483.6,
         486.8, 491.6, 501.8, 510.9, 513.0, 534.4, 538.2, ],
        [18.0, 20.5, 37.9, 89.0, 94.5, 99.4, 112.3, 116.7, 131.1, 138.5, 146.4, 153.9, 183.0, 207.9, 209.7, 263.3,
         296.8, 311.1, 319.5, 328.8, 347.8, 379.8, 406.5, 416.9, 423.3, 431.2, 434.8, 440.9, 452.6, 458.9, 480.4, 482.5,
         485.4, 490.6, 499.4, 510.0, 511.9, 533.4, 537.9, 543.5, ],
        ["speaker_SPEAKER_01", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00",
         "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_00",
         "speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_00",
         "speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_01",
         "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_00",
         "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_01",
         "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00",
         "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_01", ]
    ),
    "dilma_2.wav": (
        [5.2, 15.5, 28.2, 48.1, 50.6, 52.3, 55.7, 56.7, 63.9, 77.2, 77.3, 83.3, 93.8, 96.9, 116.7, 151.2, 154.9, 158.2,
         171.0, 174.8, 181.4, 185.3, ],
        [15.5, 27.5, 47.3, 49.8, 51.6, 55.0, 56.0, 61.8, 76.6, 82.4, 79.5, 93.1, 96.3, 115.3, 150.2, 153.8, 156.6,
         171.4, 173.6, 180.7, 183.7, 198.0, ],
        ["speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_01", "speaker_SPEAKER_01", "speaker_SPEAKER_01",
         "speaker_SPEAKER_01", "speaker_SPEAKER_01", "speaker_SPEAKER_01", "speaker_SPEAKER_01", "speaker_SPEAKER_01",
         "speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_01", "speaker_SPEAKER_01", "speaker_SPEAKER_01",
         "speaker_SPEAKER_01", "speaker_SPEAKER_01", "speaker_SPEAKER_00", "speaker_SPEAKER_01", "speaker_SPEAKER_01",
         "speaker_SPEAKER_01", "speaker_SPEAKER_01", ]
    ),
    "chester_vocals_only.wav": (
        [28.0, 187.2, 480.0],
        [142.79, 376.2, 611],
        ["speaker_SPEAKER_00", "speaker_SPEAKER_00", "speaker_SPEAKER_00"]

    ),
}


def speaker_diarization():
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token="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"
    )

    # diarization = pipeline(os.path.join(PATH, 'rawset', 'dilma_1.wav'))
    # diarization = pipeline(os.path.join(PATH, 'rawset', 'dilma_2.wav'))
    # diarization = pipeline(os.path.join(PATH, 'rawset', 'lula_podpah.wav'))
    diarization = pipeline(os.path.join(PATH, 'rawset', 'chester_vocals_only.wav'))

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")


def speakers_split():
    for key, value in VAD_RESULTS.items():
        for speaker in set(value[2]):
            print(key, speaker)
            wave, sr = torchaudio.load(os.path.join(PATH, "rawset", key))

            ends = np.array(value[1])
            starts = np.array(value[0])
            speakers = np.array(value[2])
            sp_indices = np.array([i for i, x in enumerate(speakers) if x == speaker])

            ends = ends[sp_indices]
            starts = starts[sp_indices]
            # speakers = speakers[sp_indices]

            folder_name = os.path.join(PATH, "rawset", key.split('.')[0], speaker)
            os.makedirs(folder_name, exist_ok=True)

            total = []
            for start, end in zip(starts, ends):
                total.append(((end-start) / 60))
                _subsample = wave[:, math.floor(start*sr):math.floor(end*sr)]
                torchaudio.save(
                    os.path.join(
                        folder_name,
                        key.split('.')[0] + str(start).replace('.', '_') + '_' + str(end).replace('.', '_') + '.wav'
                    ),
                    _subsample,
                    sr
                )

            print(sum(total))
            # print(wave.shape, sr, wave.shape[1] / sr)


if __name__ == '__main__':
    speakers_split()

    print('so far so good')
