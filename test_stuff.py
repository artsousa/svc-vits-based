import torchaudio

if __name__ == '__main__':
    wav, sr = torchaudio.load('dataset/44k/chester/chester_vocals_only28_0_142_79_0.000_4.200.wav')
    print(wav.shape, sr)