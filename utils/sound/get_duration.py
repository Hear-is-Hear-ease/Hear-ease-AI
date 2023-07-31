import pandas as pd
import librosa
from constant.os import *


def get_duration(paths: pd.Series) -> pd.Series:
    return librosa.get_duration(path=paths)


def print_csv_path():
    print(csv_path)
