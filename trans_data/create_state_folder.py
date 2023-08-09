from trans_data.get_state_list import get_state_list_from_dir_name
from trans_data.bit_sampling import *
from utils.sound import *
from utils.os import *
import os
import pandas as pd


def create_empty_state_folder(path: str, state_list=None) -> list[str]:
    """
    folder_path 경로에 state_list 안에 있는 state 폴더들을 생성한다.

    Parameters:
        * path: 폴더를 생성하고자 하는 경로. 경로가 존재하지 않을 경우 생성하며 존재할 경우 에러를 발생시킨다.
        * state_list: 폴더를 생성하고자 하는 state 리스트. 없을 경우 스스로 가져온다.

    Returns:
        * 생성된 state들에 따른 경로를 반환한다.
    """

    if os.path.exists(path):
        raise OSError(f'path {path} already exist.')
    else:
        os.makedirs(path)

    if state_list == None:
        state_list = get_state_list_from_dir_name(data_path)
    else:
        for state in state_list:
            if not os.path.exists(os.path.join(data_path, state)):
                raise OSError(
                    f"The path corresponding to state '{state}' does not exist.")

    if state_list == None or len(state_list) == 0:
        raise Exception(f'Can not get state list from data path {data_path}')

    state_path_list = []
    for state in state_list:
        state_path = os.path.join(path, state)
        state_path_list.append(state_path)
        os.makedirs(state_path)

    return state_path_list


def create_state_folder(data_path: str, csv_path: str, output_path: Optional[str] = None):
    """
    원본 데이터와 정보가 담긴 csv 파일을 받아 state 이름의 폴더로 구분한다.

    Parameters:
        * data_path : 원본 데이터 경로
        * csv_path  : 데이터에 대한 정보가 담긴 csv 파일
        * output_path : output_path를 받을 경우 원본 데이터를 수정하는 것이 아닌 사본을 생성한다.

    Returns: None
    """
    # path set
    main_path = data_path.rsplit('/', 1)[0]
    origin_data_path = data_path
    temp_path = (output_path if output_path != None
                 else os.path.join(main_path, '_temp'))

    # 작업할 폴더를 생성한다. 만약 _temp 라는 폴더가 존재한다면 에러가 발생된다.
    if os.path.exists(temp_path):
        raise OSError('path _temp already exists.')
    os.makedirs(temp_path)

    # Load csv
    df = pd.read_csv(csv_path, index_col=0)

    # 상위 7개의 state만 추출한다. 목표로 하는 state는 다음과 같다.
    # ['sad', 'hungry', 'hug', 'awake', 'sleepy', 'uncomfortable', 'diaper']
    top7_label = df.state.value_counts().keys().tolist()[:7]

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
        # if not is_same_sample_rate([os.path.join(path, file) for path, file in file_itorator(temp_path, include='.wav')], 16000):
        raise Exception('Resampling failed for an unknown reason.')

    # 라벨에 따른 폴더 생성
    for folder in top7_label:
        os.makedirs(os.path.join(temp_path, folder))
    os.makedirs(os.path.join(temp_path, 'etc'))

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
    # df[df['istop'] == 'etc'][['state', 'file']].to_csv(
    #     os.path.join(temp_path, 'etc.csv'))
    # df[df['istop'] != 'etc'][['state', 'file']].to_csv(
    #     os.path.join(temp_path, 'top7.csv'))
    # df = df.drop(columns='istop').to_csv(os.path.join(temp_path, 'info.csv'))

    if output_path == None:
        remove_path_with_files(data_path)
        rename(temp_path, data_path)


if __name__ == '__main__':
    main_path = os.path.join(os.getcwd().rsplit(
        'baby-cry-classification')[0], 'baby-cry-classification')
    data_path = os.path.join(main_path, 'data')
    csv_path = os.path.join(main_path, 'origin_data_info.csv')

    create_state_folder()
