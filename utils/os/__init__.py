from .move import *
from .itorate import *
from .remove import *
from .copy import *
from .rename import *


def get_state_list_from_dir_name(data_path: str, with_path=False, include_etc=False) -> list[str]:
    """
    폴더 이름으로부터 state list를 추출한다.

    Parameters:
      data_path : state list를 추출할 폴더 경로

      with_path=False : True일 경우 state 폴더의 경로를 가져온다.

      include_etc=False : True일 경우 etc 폴더를 state에 포함한다.

    Returns:
      state 리스트를 반환한다.
    """
    state_list = []
    for dir in os.listdir(data_path):
        dir_path = os.path.join(data_path, dir)
        if os.path.isdir(dir_path):
            state_list.append(dir_path if with_path else dir)

    etc_path = os.path.join(data_path, 'etc')
    if include_etc == False and os.path.exists(etc_path):
        state_list.remove(etc_path if with_path else 'etc')
    return state_list
