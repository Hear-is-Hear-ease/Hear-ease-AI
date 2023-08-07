from sklearn.metrics.pairwise import cosine_similarity
import librosa
import numpy as np
import os


def compute_melspectrogram(file_path):
    """Compute the mel spectrogram of a given WAV file."""
    y, sr = librosa.load(file_path)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    return mels


def find_most_similar_files(file_list, top_n=None):
    """
    Given a list of WAV file paths, find the m most similar files based on their mel spectrograms.

    Parameters:
    * file_list: List of paths to WAV files to analyze.
    * m: Number of most similar files to return.

    Returns:
    * List of m most similar file paths.
    """

    # 1. Compute mel spectrograms for all the files
    melspectrograms = {file_path: compute_melspectrogram(
        file_path) for file_path in file_list}

    # 2. Compute pairwise cosine similarities
    n = len(file_list)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mels1 = melspectrograms[file_list[i]].flatten()
            mels2 = melspectrograms[file_list[j]].flatten()

            # Reshape to match sizes (in case they are different)
            min_size = min(mels1.shape[0], mels2.shape[0])
            mels1 = mels1[:min_size]
            mels2 = mels2[:min_size]

            similarity_matrix[i, j] = cosine_similarity(
                mels1.reshape(1, -1), mels2.reshape(1, -1))

    # 3. Find the m most similar files
    np.fill_diagonal(similarity_matrix, 0)
    indices = np.argsort(np.sum(similarity_matrix, axis=1))

    if top_n == None:
        return [file_list[i] for i in indices[-top_n:]]

    return [file_list[i] for i in indices]
