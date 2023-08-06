import os
from typing import Optional

# import sys
# sys.path.append('/Users/jaewone/developer/tensorflow/baby-cry-classification')

from utils.os import rename_by_keyword, get_state_list_from_dir_name


def rename_files_by_state(data_path: str, state_list: Optional[list[str]] = None, include_etc=False):
    """
    state에 따라 파일의 이름을 변경한다.

    Parameters:
        data_path : 파일의 경로

        state_list=None : state 리스트를 받을 경우 state_list가 포함하는 state 폴더의 파일들만 이름을 변경한다.

        include_etc=False : True일 경우 etc 폴더를 state에 포함한다.

    Returns: None
    """

    if not os.path.exists(data_path):
        raise OSError(f'data path {data_path} not exist.')

    # Get state list if not exist
    if state_list == None:
        state_list = get_state_list_from_dir_name(
            data_path, include_etc=include_etc)
    else:
        for state in state_list:
            if not os.path.exists(os.path.join(data_path, state)):
                raise OSError(
                    f"The path corresponding to state '{state}' does not exist.")

    # rename files
    for state in state_list:
        state_path = os.path.join(data_path, state)
        file_path_list = [os.path.join(state_path, file)
                          for file in os.listdir(state_path)]
        rename_by_keyword(file_path_list, state)


if __name__ == '__main__':
    from constant.os import *

    data_path = data_path
    rename_files_by_state(data_path)
    print("Done")
