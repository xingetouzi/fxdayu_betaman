from enum import Enum

from six import with_metaclass
import abc
import os
import pathlib
import pandas as pd


class AbstractLoader(with_metaclass(abc.ABCMeta)):
    @abc.abstractmethod
    def load(self):
        raise NotImplementedError


class FileLoader(AbstractLoader):
    class FILE_TYPE(Enum):
        CSV = "csv"
        EXCEL = "excel"

    file_suffix = {
        FILE_TYPE.CSV: "csv",
        FILE_TYPE.EXCEL: ("xlsx", "xls")
    }

    def __init__(self, path, type=None):
        self._type = type
        path = pathlib.Path(path)
        if not path.is_absolute():
            self._path = pathlib.Path(os.getcwd()) / path
        else:
            self._path = path

    def load_csv(self):
        return pd.read_csv(str(self._path))

    def load_excel(self):
        return pd.read_excel(str(self._path))

    def load(self):
        if not self._type:
            suffix = self._path.name.split(".")[-1]
            for t, s in self.file_suffix.items():
                if suffix in s:
                    return getattr(self, "load_" + t.value)()
        else:
            if isinstance(self._type, self.FILE_TYPE):
                func = getattr(self, "load_" + self._type.value, None)
            else:
                func = getattr(self, "load_" + str(self._type), None)
            if func:
                return func()
        raise RuntimeError("Unsupported file type: {}".format(self._type))
