import os
from typing import Union, Optional

from utils.os.itorate import file_itorator


# remove file
def remove_file(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)


# remove all files in path
def remove_files(path: str,
                 include: Optional[Union[str, list[str]]] = None,
                 exclude: Optional[Union[str, list[str]]] = None):
    for path, file in file_itorator(path, include, exclude):
        os.remove(os.path.join(path, file))
