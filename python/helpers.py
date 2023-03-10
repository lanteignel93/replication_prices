from dataclasses import dataclass, fields
from pathlib import Path

from dynamics import StockDynamics, StockDynamicsType, JumpDiffusionProcess, GeometricBrownianMotion


def dataclass_from_json(dataclass: dataclass, json_path: str | Path):
    field_set = {f.name for f in fields(dataclass) if f.init}
    with open(json_path, "r") as f:
        data = json.load(f)

    filteredArgDict = {k: v for k, v in data.items() if k in field_set}
    return dataclass(**filteredArgDict)


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


def get_stock_dynamcis(
    
):
    pass