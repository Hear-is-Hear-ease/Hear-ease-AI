import os
from uuid import uuid4


def rename(from_path: str, to_path: str):
    """
    이름을 변경한다.

    Parameters:
      from_path: 변경하고자 하는 파일 이름
      to_path:   변경될 파일 이름

    Returns: None
    """
    os.rename(from_path, to_path)


def rename_by_keyword(file_list: list[str], keyword: str) -> list[str]:
    """
    키워드를 파일 이름으로 가지도록 파일명을 변경한다. 예: label_1, label_2, ...

    Parameters:
      file_list: 이름을 변경할 파일의 경로 리스트
      keyword: 이름 변경에 사용될 키워드

    Returns: 변환된 파일 경로 리스트
    """

    # 중복된 파일명이 존재할 경우 os.rename 함수는 두 항목 모두 삭제한다.
    # 이에 uuid4를 이용하여 임의의 이름으로 변경하여 중복을 방지한 다음 rename을 수행한다.

    random_file_list = []
    for i in range(len(file_list)):
        path = file_list[i].rsplit('/', 1)[0]
        ex = file_list[i].rsplit('.', 1)[1]
        uuid_file_path = f'{path}/{uuid4()}.{ex}'
        random_file_list.append(uuid_file_path)
        os.rename(file_list[i], uuid_file_path)

    renamed_file_list = []
    for i in range(len(random_file_list)):
        path = random_file_list[i].rsplit('/', 1)[0]
        ex = random_file_list[i].rsplit('.', 1)[1]
        renamed_file = f'{path}/{keyword}_{i+1}.{ex}'
        renamed_file_list.append(renamed_file)
        os.rename(random_file_list[i], renamed_file)
    return renamed_file_list


if __name__ == '__main__':
    from constant.os import *
    from utils.os import rename_by_keyword, rename
    import pandas as pd

    df = pd.read_csv('sample_data.csv', index_col=0)

    # df['new_file'] = ''
    # for state in df.state.unique():
    #     target = df['state'] == state
    #     print(np.array([i + 1 for i in range(len(df[df.state == state]))]).astype(str))
    #     df.loc[target,'new_file'] = df['state'] + np.array([i + 1 for i in range(len(df[df.state == state]))]).astype(str)

    # df
    df = df.assign(cumcount=(df.groupby('state').cumcount() + 1).astype(str))
    df = df.assign(ex=df['file'].apply(lambda df: df.rsplit('.', 1)[1]))
    df = df.assign(new_file=df.state + '_' + df.cumcount + '.' + df.ex)

    # change file name
    df[['file', 'new_file']].apply(lambda df: rename(
        os.path.join(main_path, 'sample_data', df['file']),
        os.path.join(main_path, 'sample_data', df['new_file'])
    ), axis=1)

    # drop new file column
    df = df.drop(columns=['cumcount', 'ex', 'file'])
    df = df.rename(columns={'new_file': 'file'})
    df.tail(3)
