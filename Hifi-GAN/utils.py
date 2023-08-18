import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt


# 스펙트로그램(spectrogram)을 시각화하기 위한 함수
def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


# 신경망 모델의 가중치를 초기화
### 주어진 모듈 m의 클래스 이름을 확인하고, 클래스 이름에 "Conv" 문자열이 포함되어 있으면 해당 모듈의 가중치를 평균 mean과 표준편차 std를 가지고 정규분포를 따르는 값으로 초기화
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# 모듈 m에 가중치 정규화(weight normalization)를 적용
def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


# 커널 크기(kernel_size)와 팽창 비율(dilation)을 받아서 필요한 패딩 값을 계산 <컨볼루션 계산 시 패딩 필요>
def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


# 모델 체크포인트를 불러오기
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


#  모델 체크포인트를 저장
def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


# 특정 디렉토리에서 특정 접두사(prefix)를 가진 체크포인트 파일을 검색
### 주어진 디렉토리 cp_dir 내에서 prefix로 시작하고 그 뒤에 8자리의 숫자가 올 수 있는 파일을 검색하여 리스트로 반환, 이 리스트에 체크포인트 파일이 없을 경우 None을 반환하고, 있을 경우 가장 최근의 체크포인트 파일을 반환
def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]