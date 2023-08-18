import os
import shutil


# 딕셔너리의 값을 마치 객체의 속성처럼 접근
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# 주어진 config 파일을 새 경로로 복사하고, 필요한 디렉토리를 생성하는 역할
def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))