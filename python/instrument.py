from dataclasses import dataclass
from enum import Enum


class OptionType(Enum):
    PUT = 0
    CALL = 1


class ExerciseType(Enum):
    EUROPEAN = 0
    AMERICAN = 1


class HedgeType(Enum):
    REALIZED = 0
    IMPLIED = 1


@dataclass
class EquityOption:
    type: OptionType
    exercise: ExerciseType
    spot: float
    strike: float
    vol: float
    drift: float
    days_to_maturity: int
