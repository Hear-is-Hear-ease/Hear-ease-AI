from bit_sampling import resampling, is_same_sample_rate
from trans_data import get_sample_rate
import sys
import os
import numpy as np
import pandas as pd

from utils.os import *
from utils.sound import *
from constant.os import *
sys.path.append('/Users/jaewone/developer/tensorflow/baby-cry-classification')


def create_state_folder():
    # path set
    origin_data_path = data_path
    temp_path = os.path.join(main_path, '_temp')
    csv_path = csv_path

    # 작업할 폴더를 생성한다. 만약 _temp 라는 폴더가 존재한다면 에러가 발생된다.
    if os.path.exists(temp_path):
        raise OSError(f'path {temp_path} already exists.')
    os.makedirs(temp_path)

    # Load csv
    df = pd.read_csv(csv_path, index_col=0)

    # 상위 7개의 state만 추출한다. 목표로 하는 state는 다음과 같다.
    # ['sad', 'hungry', 'hug', 'awake', 'sleepy', 'uncomfortable', 'diaper']
    top7_label = df.state.value_counts().keys().tolist()[:7]

    # 라벨에 따른 폴더 생성
    for folder in top7_label:
        os.makedirs(os.path.join(temp_path, folder))
    os.makedirs(os.path.join(temp_path, 'etc'))

    # 음성의 sample rate를 추출한 origin_sample_rate 열을 추가한다.
    df = df.assign(origin_sample_rate=df['file'].apply(
        lambda file: get_sample_rate(os.path.join(origin_data_path, file))))

    # 에러가 난 파일은 na 값을 가지기에 데이터프레임에서 제외시킨다.
    print(
        f'전체 {len(df)}개의 데이터 중 사용할 수 없는 {(len(df[df.origin_sample_rate.isna()]))}개의 데이터가 제외되었다.')

    df = df[df.origin_sample_rate.notna()]

    # Sampling rate를 통일한다.
    resampling(
        file_path_list=[os.path.join(origin_data_path, file)
                        for file in df.file.tolist()],
        output_path=temp_path,
        target_sample_rate=16000
    )

    if not is_same_sample_rate([os.path.join(temp_path, file) for file in os.listdir(temp_path)], 16000):
        raise Exception('Resampling failed for an unknown reason.')

    # 각 라벨에 맞는 폴더로 이동

    def move_file_by_label(ser):
        file, state = ser
        move_file(
            os.path.join(temp_path, file),
            os.path.join(temp_path, state, file))

    df = df.assign(istop=df.state.where(df.state.isin(top7_label), 'etc'))
    df[['file', 'istop']].apply(lambda ser: move_file_by_label(ser), axis=1)

    # 에러가 난 파일을 제외한 다음 원본 csv를 수정한다.
    # 또한 상위 7개의 state만을 가지는 csv 파일인 top7.csv와 나머지 state들을 가지는 etc.csv 파일을 생성한다.
    df[df['istop'] == 'etc'][['state', 'file']].to_csv(
        os.path.join(temp_path, 'etc.csv'))
    df[df['istop'] != 'etc'][['state', 'file']].to_csv(
        os.path.join(temp_path, 'top7.csv'))
    df = df.drop(columns='istop').to_csv(os.path.join(temp_path, 'info.csv'))

    # 경고 메세지를 출력
    print('state에 따른 폴더 생성 및 resampling 완료.')

    print('미연의 사고를 방지하기 위해 원본 데이터인 data 폴더를 삭제하는 것은 사용자에게 맡긴다.')

    print('data 폴더를 삭제한 다음 _temp 폴더의 이름을 data로 변형하여 사용하면 된다.')


if __name__ == '__main__':
    create_state_folder()
