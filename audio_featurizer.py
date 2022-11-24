import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import pickle


def audio_process(songname: str) -> pd.DataFrame:
    """
    :rtype: DataFrame of all the features
    """

    audio_file, sr = librosa.load(songname, mono=True, duration=30)

    # rmse = librosa.feature.rms(y=y)
    # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    # spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    # spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    # rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # zcr = librosa.feature.zero_crossing_rate(y)
    # mfcc = librosa.feature.mfcc(y=y, sr=sr)

    x = []

    # Chromogram
    # Increase or decrease hop_length to change how granular you want your data to be
    hop_length = 5000
    chromagram = librosa.feature.chroma_stft(audio_file, sr=sr, hop_length=hop_length)
    x.append(chromagram.mean())
    x.append(chromagram.var())

    # rms
    rms = librosa.feature.rms(audio_file)

    x.append(rms.mean())
    x.append(rms.var())

    # spectral_centroids
    spectral_centroids = librosa.feature.spectral_centroid(audio_file, sr)
    x.append(spectral_centroids.mean())
    x.append(spectral_centroids.var())

    # spectral_bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(audio_file, sr)
    x.append(spec_bw.mean())
    x.append(spec_bw.var())

    # spectral_rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(audio_file, sr)
    x.append(spectral_rolloff.mean())
    x.append(spectral_rolloff.var())

    # zero_crossing_rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_file, sr)
    x.append(zero_crossing_rate.mean())
    x.append(zero_crossing_rate.var())

    # harmony
    y_harm, y_perc = librosa.effects.hpss(audio_file)

    x.append(y_harm.mean())
    x.append(y_harm.var())

    # perceptual_weighting
    # perceptr = librosa.perceptual_weighting(audio_file, sr)
    # x.append(perceptr.mean())
    # x.append(perceptr.var())

    # tempo
    tempo, beat_times = librosa.beat.beat_track(audio_file, sr, units='time')
    x.append(tempo)



    vector = pd.DataFrame ([x], columns = [['chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
       'spectral_centroid_mean', 'spectral_centroid_var',
       'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
       'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
       'harmony_mean', 'harmony_var', 'tempo']])
    return vector


def spectrogram_plot(audio_file: str):
    """
    :rtype: Plot
    """
    y, sr = librosa.load(audio_file, mono=True, duration=5)
    plot = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
    plot = librosa.power_to_db(plot, ref=np.max)
    librosa.display.specshow(plot, y_axis='mel', x_axis='time')
    plt.title('Mel-frequency spectrogram')
    plt.colorbar()
    plt.tight_layout()


def soundwaves_plot(audio_file: str):
    """
    :rtype: Plot
    """
    y, sr = librosa.load(audio_file, mono=True, duration=5)



    plt.figure(figsize=(16, 6))
    librosa.display.waveshow(y=y, sr=sr, color="#A300F9")
    plt.title("Sound Waves", fontsize=23)
    #plt.colorbar()
    #plt.tight_layout()


def model1(atributes):
    pickled_model = pickle.load(open('tree.pkl', 'rb'))
    return pickled_model.predict(atributes)


def model2(atributes):
    pickled_model = pickle.load(open('xgb.pkl', 'rb'))
    return pickled_model.predict(atributes)

