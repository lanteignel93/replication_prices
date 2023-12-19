import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import abstractmethod
from typing import Tuple
from enum import Enum
from instrument import (
    EquityOption, 
    OptionType,
    ExerciseType
)



class StockDynamicsType(Enum):
    GEOMETRIC_BROWNIAN_MOTION = 0
    JUMP_DIFFUSION_PROCESS = 1


@dataclass
class MarketDynamics:
    risk_free_rate: float


@dataclass
class StockDynamicsParameters:
    # stock_dynamics_type: StockDynamicsType
    option: EquityOption
    num_steps: int


@dataclass
class GeometricBrownianMotionParameters(StockDynamicsParameters):
    stock_dynamics_type = StockDynamicsType.GEOMETRIC_BROWNIAN_MOTION


@dataclass
class JumpDiffusionProcessParameters(StockDynamicsParameters):
    stock_dynamics_type = StockDynamicsType.JUMP_DIFFUSION_PROCESS
    lambd: float
    m: float
    nu: float


class StockDynamics:
    def __init__(self, parameters: StockDynamicsParameters, seed: int = 0) -> None:
        self._parameters = parameters
        self._init_params()
        self._seed = seed
        self._name = None

    @abstractmethod
    def generate_stock_path(self):
        pass

    @abstractmethod
    def _init_params(self):
        pass


class GeometricBrownianMotion(StockDynamics):
    def __init__(
        self, parameters: GeometricBrownianMotionParameters, seed: float = None
    ) -> None:
        super().__init__(parameters, seed)
        self._parameters = parameters
        self._seed = seed
        self._init_params()

    def generate_stock_path(self) -> Tuple[np.array, pd.DataFrame]:
        """
        This function is used to generate the stock path under the
        geometric brownian motion dynamics

        Returns:
            Tuple[np.array, pd.DataFrame]: tuple with first element
            being a np.array with the stock prices and 2nd element of a
            dataframe with the different random variable generated to
            compute the stock path
        """
        # Time step size
        dt = self._option.days_to_maturity / (252 * self._num_steps)
        if self._seed is not None:
            np.random.seed(seed=self._seed + 1234)
        #
        random_path = np.random.normal(0, np.sqrt(dt), size=self._num_steps)

        stock_path = (
            self._option.spot
            * np.exp(
                (self._option.drift - 0.5 * self._option.vol**2) * dt
                + self._option.vol * random_path
            ).cumprod()
        )
        stock_path = np.append(self._option.spot, stock_path)
        random_path = np.append(random_path, 0)

        random_path = pd.DataFrame({"geo_Z": random_path})
        return stock_path, random_path

    def _init_params(self):
        """
        This function inits the class via the parameters
        """
        self._option = self._parameters.option
        self._num_steps = self._parameters.num_steps
        self._type = self._parameters.stock_dynamics_type
        self._name = self._type.name.replace("_", " ").title()


class JumpDiffusionProcess(StockDynamics):
    def __init__(
        self, parameters: GeometricBrownianMotionParameters, seed: float = 0
    ) -> None:
        super().__init__(parameters, seed)
        self._parameters = parameters
        self._seed = seed
        self._init_params()

    def generate_stock_path(self) -> Tuple[np.array, pd.DataFrame]:
        """
        This function is used to generate the stock path under the
        jump diffusion process

        Returns:
            Tuple[np.array, pd.DataFrame]: tuple with first element
            being a np.array with the stock prices and 2nd element of a
            dataframe with the different random variable generated to
            compute the stock path
        """
        # Time step size
        dt = self._option.days_to_maturity / (252 * self._num_steps)
        np.random.seed(seed=self._seed + 1234)

        random_norm_path = np.random.normal(
            0, self._option.vol * np.sqrt(dt), size=self._num_steps
        )
        random_norm_poisson_path = np.random.normal(
            self._m, self._nu, size=self._num_steps
        )
        random_poisson_jumps = np.random.poisson(
            self._lambda * dt, size=self._num_steps
        )

        poisson_random_v = np.multiply(
            random_poisson_jumps, random_norm_poisson_path
        ).cumsum(axis=0)
        geometric_path = np.cumsum(
            (
                (
                    self._option.drift
                    - 0.5 * self._option.vol**2
                    - self._lambda * (self._m + 0.5 * self._nu**2)
                )
                * dt
                + random_norm_path
            ),
            axis=0,
        )

        stock_path = np.append(
            self._option.spot,
            self._option.spot * np.exp(poisson_random_v + geometric_path),
        )

        random_norm_path = np.append(random_norm_path, 0)
        random_norm_poisson_path = np.append(random_norm_poisson_path, 0)
        random_poisson_jumps = np.append(random_poisson_jumps, 0)

        random_df = pd.DataFrame(
            {
                "geo_Z": random_norm_path,
                "geo_poisson": random_norm_poisson_path,
                "jump_poisson": random_poisson_jumps,
            }
        )

        return stock_path, random_df

    def _init_params(self):
        self._option = self._parameters.option
        self._num_steps = self._parameters.num_steps
        self._lambda = self._parameters.lambd
        self._m = self._parameters.m
        self._nu = self._parameters.nu
        self._type = self._parameters.stock_dynamics_type
        self._name = self._type.name.replace("_", " ").title()


# BELOW IS DEMO ON HOW TO GENERATE SOME PATHS
if __name__ == "__main__":
    S = 100
    DAYS = 21
    SIGMA_REALIZED = 0.3
    SIGMA_IMPLIED = 0.4
    R = 0.04
    MU = 0.12
    NUM_PATHS = 10
    NUM_SIMS = 100

    option_realized = EquityOption(
        type=OptionType.CALL,
        exercise=ExerciseType.EUROPEAN,
        spot=S,
        strike=120,
        vol=SIGMA_REALIZED,
        drift=MU,
        days_to_maturity=DAYS,
    )

    jump_parameters = JumpDiffusionProcessParameters(
        option=option_realized, num_steps=NUM_SIMS, lambd=5, m=0.0, nu=0.12
    )

    geo_parameters = GeometricBrownianMotionParameters(
        option=option_realized, num_steps=NUM_SIMS
    )

    stock_dynamics = JumpDiffusionProcess(parameters=jump_parameters, seed=0)

    S1, Z1 = stock_dynamics.generate_stock_path()

    stock_dynamics = GeometricBrownianMotion(parameters=geo_parameters, seed=0)

    S2, Z2 = stock_dynamics.generate_stock_path()
