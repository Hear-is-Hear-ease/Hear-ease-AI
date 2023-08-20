from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator


h = None
device = None


# 체크포인트 파일을 로드
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


# 멜 스펙트로그램을 생성
def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


# 지정된 디렉토리에서 특정 접두사를 가진 체크포인트 파일들 중 가장 최근 파일을 찾아 반환
def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


# 음성 합성을 수행
def inference(a):
    # 생성기 모델을 정의
    generator = Generator(h).to(device)

    # 생성기 모델의 상태 사전 가져옴
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    # 입력 WAV 파일 리스트 로드
    filelist = os.listdir(a.input_wavs_dir)

    # 출력 디렉토리 생성
    os.makedirs(a.output_dir, exist_ok=True)

    # 모델 설정
    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            # WAV 파일 로드 및 정규화
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)

            # 멜 스펙트로그램 생성
            x = get_mel(wav.unsqueeze(0))

            # 모델에 입력 전달하여 음성 생성
            y_g_hat = generator(x)

            # 음성 후처리
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            # 생성된 음성을 WAV 파일로 저장
            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


#  음성 합성 모델을 사용하여 실제 음성을 생성하고 저장 (멜 스펙트로그램을 입력으로 받아)
def main():
    # 프로세스 초기화 메시지 출력
    print('Initializing Inference Process..')

    # 명령줄 인자 파싱을 위한 ArgumentParser 객체 생성
    parser = argparse.ArgumentParser()

    # 인자 추가: 입력 WAV 파일 디렉토리, 출력 디렉토리, 체크포인트 파일
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    # 설정 파일 로드를 위한 경로 생성
    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')

    # 설정 파일 읽기
    with open(config_file) as f:
        data = f.read()

    # 전역 설정 객체 생성 및 로드된 설정 값 적용
    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    # 난수 생성 시드 설정
    torch.manual_seed(h.seed)

    # 디바이스 설정: GPU 사용 가능 시 GPU로, 그렇지 않으면 CPU로 설정
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # inference 함수 호출
    inference(a)

# 스크립트가 직접 실행될 때만 main 함수 호출
if __name__ == '__main__':
    main()