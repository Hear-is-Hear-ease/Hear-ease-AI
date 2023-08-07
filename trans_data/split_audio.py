import numpy as np
import os
import librosa
from scipy.io import wavfile

# import sys
# sys.path.append('/Users/jaewone/developer/tensorflow/baby-cry-classification')

from constant.os import *


# 각 파일의 2초부터 7초 사이의 음성을 추출한다.
def split_audios(file_list: list[str],
                 output_path: str,
                 start_time: float,
                 end_time: float):
    """
    파일 리스트의 음성 파일들을 start_time부터 end_time까지 자른다.

    Parameters:

      file_list : 변환할 파일의 경로 리스트. wav 파일만 가능하다.

      output_path : 변환된 파일을 저장할 폴더 경로

      start_time : 자르기를 시작하는 시점(초)

      end_time : 자르기를 종료하는 시점(초)

    """

    # output_path 경로(폴더)가 없을 경우 생성한다.
    if os.path.exists(output_path):
        raise OSError(f'output path {output_path} already exists.')
    else:
        os.makedirs(output_path)

    # 파일을 자른다.
    for file_path in file_list:

        # 파일의 정보를 읽어온다.
        y, sr = librosa.load(file_path, sr=None)

        # 음성 자르기
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segmented_audio = y[start_sample:end_sample]

        # 저장
        segmented_audio = np.array(segmented_audio * (2**15), dtype=np.int16)
        wavfile.write(os.path.join(
            output_path, file_path.rsplit('/', 1)[1]), sr, segmented_audio)


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv(data_path, 'top7.csv', index_col=0)

    split_audios(
        file_list=[os.path.join(data_path, file)
                   for file in df['file'].tolist()],
        from_path=data_path,
        output_path=os.path.join(main_path, 'trin_audio_data'),
        start_time=2,
        end_time=7
    )
