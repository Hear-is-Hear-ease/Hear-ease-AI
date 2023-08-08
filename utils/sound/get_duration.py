import pandas as pd
import librosa
from constant.os import *


# Get duration of sound file.
def get_duration(paths: str) -> float:
    return librosa.get_duration(path=paths)
