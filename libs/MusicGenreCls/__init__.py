import math
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter


def get_mfccs(audio_list, fs=22050, duration=30, n_fft=2048, hop_length=512, n_mfcc=13, num_segments=10):
    samples_per_track = fs * duration
    samps_per_segment = int(samples_per_track / num_segments)
    mfccs_per_segment = math.ceil(samps_per_segment / hop_length)
    mfcc_list = []
    for audio in audio_list:
        for seg in range(num_segments):
            start_sample = seg * samps_per_segment
            end_sample = start_sample + samps_per_segment
            mfcc = librosa.feature.mfcc(
                y=audio[start_sample:end_sample],
                sr=fs,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mfcc=n_mfcc
            )
            mfcc = mfcc.T
            if len(mfcc) == mfccs_per_segment:
                mfcc_list.append(mfcc.tolist())
    return np.array(mfcc_list)


def make_prediction(audio_list, label_list):
    X = get_mfccs(audio_list)[..., np.newaxis]
    num_segments = len(X) // len(audio_list)
    model = load_model('./libs/MusicGenreCls/model_cnn3.h5')
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    pred_list = model.predict(X)
    # pred_list = np.argmax(pred_list, axis=1)
    # genre_list = []
    # for i in range(len(audio_list)):
    #     counter = Counter(pred_list[i * num_segments: i * num_segments + num_segments])
    #     genre_list.append(genres[counter.most_common()[0][0]])
    # return genre_list

    prob_list = []
    for i in range(0, len(pred_list), num_segments):
        label = genres.index(label_list[i // num_segments])
        prob = pred_list[i: i + num_segments, label].mean()
        prob_list.append(float(prob))
    return prob_list
