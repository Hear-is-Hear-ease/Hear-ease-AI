import wave
import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, resample_poly
import soundfile as sf
import os
import librosa

from constant.os import *


def get_wav_info(file_path):
    # Read the input WAV file
    with wave.open(file_path, 'rb') as wav_in:
        sample_rate = wav_in.getframerate()
        num_channels = wav_in.getnchannels()
        sample_width = wav_in.getsampwidth()
        num_frames = wav_in.getnframes()
        audio_data = wav_in.readframes(num_frames)

        print(f'sample_rate: {sample_rate}')   # 16000
        print(f'num_channels: {num_channels}')  # 1
        print(f'sample_width: {sample_width}')  # 2
        print(f'num_frames: {num_frames}')     # 239630
        # print(f'audio_data: {audio_data}')   # a7\xed\xa6\xe2\x95\ ...


def get_sample_rate(file_path: str):
    try:
        with wave.open(file_path, 'rb') as wav_in:
            sample_rate = wav_in.getframerate()
            return sample_rate
    except:
        return np.nan


def bit_sampling(files, output_folder, target_sample_rate):
    for input_path in files:
        file = input_path.rsplit('/', 1)[1]
        output_path = os.path.join(output_folder, file)

        # Read the input WAV file
        with wave.open(input_path, 'rb') as wav_in:
            sample_rate = wav_in.getframerate()
            num_channels = wav_in.getnchannels()
            sample_width = wav_in.getsampwidth()
            num_frames = wav_in.getnframes()
            audio_data = wav_in.readframes(num_frames)

        # Convert 2-byte audio data to a 1D numpy array with dtype 'int16'
        audio_data = np.frombuffer(audio_data, dtype='int16')

        # Check if the sample rate matches the target rate
        if sample_rate != target_sample_rate:
            # Resample the audio data to the target sample rate
            # resampled_data = resample(audio_data, int(
            #     len(audio_data) * target_sample_rate / sample_rate))

            num_samples_resampled = int(
                len(audio_data) * target_sample_rate / sample_rate)
            resampled_data = resample_poly(
                audio_data, num_samples_resampled, len(audio_data))
            # Convert back to 2-byte audio data
            resampled_data = resampled_data.astype('int16')

            # Write the resampled data to the output WAV file
            with wave.open(output_path, 'wb') as wav_out:
                wav_out.setnchannels(num_channels)
                wav_out.setsampwidth(sample_width)
                wav_out.setframerate(target_sample_rate)
                wav_out.writeframes(resampled_data)

        else:
            # If the sample rate already matches, simply copy the file to the output folder
            with open(output_path, 'wb') as wav_out:
                wav_out.write(audio_data)


def down_sampling(files, output_folder, target_sample_rate):
    for input_path in files:
        file = input_path.rsplit('/', 1)[1]
        output_path = os.path.join(output_folder, file)

        data, samplerate = sf.read(input_path)
        sf.write(output_path, data, target_sample_rate, subtype='PCM_16')


def can_not_resampling_file():
    target = '/Users/jaewone/developer/tensorflow/baby-cry-classification/sample/laugh_1.m4a_68.wav'
    destinationPath = '/Users/jaewone/developer/tensorflow/baby-cry-classification/sample/aa.wav'

    buf = None

    with open(target, 'rb') as tf:
        buf = tf.read()
        buf = buf+b'0' if len(buf) % 2 else buf

    pcm_data = np.frombuffer(buf, dtype='int16')
    wav_data = librosa.util.buf_to_float(x=pcm_data, n_bytes=2)
    sf.write(destinationPath, wav_data, 16000, format='WAV',
             endian='LITTLE', subtype='PCM_16')
