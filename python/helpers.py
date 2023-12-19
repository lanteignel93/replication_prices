from dataclasses import dataclass, fields
from pathlib import Path
import json
import multiprocessing
from typing import Callable



def dataclass_from_json(dataclass: dataclass, json_path: str | Path):
    field_set = {f.name for f in fields(dataclass) if f.init}
    with open(json_path, "r") as f:
        data = json.load(f)

    filteredArgDict = {k: v for k, v in data.items() if k in field_set}
    return dataclass(**filteredArgDict)

def multiprocessing_function(func: Callable, multi_proc_func: Callable, num_cores: int, df_dict_map: dict, *args):
    res_dic = multiprocessing.Manager().dict()
    
    processes = [
        multi_proc_func(target=func, args=(df_dict_map[i], res_dic, i, *args))
        for i in range(num_cores)
    ]
    [p.start() for p in processes]
    [p.join() for p in processes]
    
    return {key: item for key, item in res_dic.items()}


class DataClassUnpack:
    classFieldCache = {}

    @classmethod
    def instantiate(cls, dataclass: dataclass, json_path: str | Path):
        with open(json_path, "r") as f:
            data = json.load(f)
        if dataclass not in cls.classFieldCache:
            cls.classFieldCache[dataclass] = {
                f.name for f in fields(dataclass) if f.init
            }

        fieldSet = cls.classFieldCache[dataclass]
        filteredArgDict = {k: v for k, v in data.items() if k in fieldSet}
        return dataclass(**filteredArgDict)
