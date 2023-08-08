import pandas as pd
import librosa
import os

main_path = os.path.join(os.getcwd().rsplit(
    'baby-cry-classification')[0], 'baby-cry-classification')
data_path = os.path.join(main_path, 'data')
csv_path = os.path.join(main_path, 'origin_data_info.csv')


# Get duration of sound file.
def get_duration(paths: str) -> float:
    return librosa.get_duration(path=paths)
