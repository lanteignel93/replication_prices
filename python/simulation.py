from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import scipy as sp
from typing import Literal

from dynamics import MarketDynamics, StockDynamics, StockDynamicsType
from instrument import EquityOption, OptionType, HedgeType


@dataclass
class SimulationParameters:
    num_steps: float
    num_paths: float


class Simulation:
    def __init__(self, parameters: SimulationParameters) -> None:
        self._parameters = parameters

    def run_replication_price_simulation(
        self,
        opt_r: EquityOption,
        opt_i: EquityOption,
        market_dynamics: MarketDynamics,
        stocks_dynamics: StockDynamics,
        hedge_vol: HedgeType,
    ) -> float:
        """
        Computes the replication price with respect to all the dynamics passed to the function

        Args:
            opt_r (EquityOption): Realized Volatility Option Characteristic
            opt_i (EquityOption): Implied Volatility Option Characteristic
            market_dynamics (MarketDynamics): Dynamics of the Market
            stocks_dynamics (StockDynamics): Stock Dynamics
            hedge_vol (HedgeType): Whether or not to Hedge with Realized Volatility or Implied Volatility

        Returns:
            float: _description_
        """
        self._dt = opt_r.days_to_maturity / (252 * self._parameters.num_steps)

        S, _ = stocks_dynamics.generate_stock_path()

        df = pd.DataFrame()
        df["S"] = S
        df["tau"] = np.linspace(
            opt_r.days_to_maturity / 252, 0, num=self._parameters.num_steps + 1
        )

        df = self._compute_greeks(df=df, opt=opt_r, mkt=market_dynamics, suffix="_r")
        df = self._compute_greeks(df=df, opt=opt_i, mkt=market_dynamics, suffix="_i")

        replication_price = self._compute_replication_price(
            df=df,
            delta_ref="delta_r"
            if hedge_vol == HedgeType.REALIZED
            else "delta_i",
            r=market_dynamics.risk_free_rate,
        )

        replication_price = replication_price * np.exp(
            -market_dynamics.risk_free_rate * opt_r.days_to_maturity / 252
        )

        implied_price, realized_price = df.bs_price_i.iloc[0], df.bs_price_r.iloc[0]
        return replication_price, implied_price, realized_price

    def run_hedge_vol_simulation(
        self,
        opt_r: EquityOption,
        opt_i: EquityOption,
        market_dynamics: MarketDynamics,
        stocks_dynamics: StockDynamics,
        hedge_vol: HedgeType,
    ) -> pd.DataFrame:
        """
        Compute the PnL of the hedged portfolio Simulation

        Args:
            opt_r (EquityOption): Realized Volatility Option Characteristic
            opt_i (EquityOption): Implied Volatility Option Characteristic
            market_dynamics (MarketDynamics): Dynamics of the Market
            stocks_dynamics (StockDynamics): Stock Dynamics
            seed (float): Seed for
            hedge_vol (HedgeType): Whether or not to Hedge with Realized Volatility or Implied Volatility

        Returns:
            pd.DataFrame: DataFrame with cumulative PnL of the strategy
        """
        self._dt = opt_r.days_to_maturity / (252 * self._parameters.num_steps)

        if stocks_dynamics._type != StockDynamicsType.GEOMETRIC_BROWNIAN_MOTION:
            return NotImplementedError(
                "Only Geometric Brownian Motion is implemented for this simulation"
            )
        else:
            S, Z = stocks_dynamics.generate_stock_path()

        df = pd.DataFrame()
        df["S"] = S
        df["Z"] = Z
        df["tau"] = np.linspace(
            opt_r.days_to_maturity / 252, 0, num=self._parameters.num_steps + 1
        )

        df = self._compute_greeks(df=df, opt=opt_r, mkt=market_dynamics, suffix="_r")
        df = self._compute_greeks(df=df, opt=opt_i, mkt=market_dynamics, suffix="_i")

        df["dPnL"] = self._replication_hedge_pnl(
            gamma=df.gamma_i,
            S=df.S,
            sigma_r=opt_r.vol,
            sigma_i=opt_i.vol,
            delta_r=df.delta_r
            if hedge_vol == HedgeType.REALIZED
            else HedgeType.IMPLIED,
            delta_i=df.delta_i,
            mu=opt_r.drift,
            r=market_dynamics.risk_free_rate,
            Z=df.Z,
        ).shift(1)
        df["PnL"] = df.dPnL.cumsum()
        df.fillna(0, inplace=True)
        return df

    def _compute_replication_price(
        self, df: pd.DataFrame, delta_ref: Literal["delta_r", "delta_i"], r: float
    ) -> float:
        """
        Computes the replication price

        Args:
            df (pd.DataFrame): DataFrame with Spot and all the greeks needed compute replication price
            delta_ref (Literal['delta_r', 'delta_i']): Reference delta to hedge
            r (float): Riskless Rate

        Returns:
            float: Replication Price
        """
        df["delta_v"] = df[delta_ref].diff(1).fillna(0)
        df["fv"] = np.exp(df.tau * r)
        df["delta_hedged_fv_pnl"] = df["delta_v"] * df.S * df.fv

        final_price = df.bs_price_i.iloc[-1]

        init_hedge = df[delta_ref].iloc[0] * df.S.iloc[0]
        final_hedge = df[delta_ref].iloc[-1] * df.S.iloc[-1]

        final_value = (
            final_price
            - final_hedge
            + init_hedge * np.exp(r * df.tau.iloc[0])
            + df["delta_hedged_fv_pnl"].sum()
        )

        return final_value

    def _d1(self, S: float, K: float, r: float, sigma: float, tau: float) -> float:
        """
        Compute's Black-Scholes d1

        Args:
            S (float): Spot Price
            K (float): Strike Price
            r (float): Riskless Rate
            sigma (float): Volatility
            tau (float): Time to expiration in years

        Returns:
            float: d1
        """
        return (np.log(S / K) + (r + 0.5 * np.power(sigma, 2)) * tau) / (
            sigma * np.sqrt(tau)
        )

    def _d2(self, d1: float, sigma: float, tau: float) -> float:
        """
        Compute's Black-Scholes d2

        Args:
            d1 (float): Black-Scholes d1
            sigma (float): Volatility
            tau (float): Time to expiration in years

        Returns:
            float: d2
        """
        return d1 - sigma * np.sqrt(tau)

    def _gamma(
        self, K: float, r: float, tau: float, d2: float, S: float, sigma: float
    ) -> float:
        """
        Compute's Black-Scholes gamma

        Args:
            K (float): Option Strike
            r (float): Riskless Rate
            tau (float): Time to expiration in years
            d2 (float): Black-Scholes d2
            S (float): Spot Price
            sigma (float): Volatility

        Returns:
            float: gamma
        """
        return (
            K
            * np.exp(-r * tau)
            * sp.stats.norm.pdf(d2)
            / (np.power(S, 2) * sigma * np.sqrt(tau))
        )

    def _delta(self, d1: float, cp_flag: OptionType) -> float:
        """
        Compute's Black-Scholes delta

        Args:
            d1 (float): Black-Scholes d1
            cp_flag (OptionType): Option Type (put/call)

        Returns:
            float: delta
        """
        return (
            sp.stats.norm.cdf(d1)
            if cp_flag == OptionType.CALL
            else -sp.stats.norm.cdf(-d1)
        )

    def _vega(self, K: float, r: float, tau: float, d2: float):
        return K * np.exp(-r * tau) * sp.stats.norm.pdf(d2) * np.sqrt(tau)

    def _theta(
        self,
        K: float,
        r: float,
        tau: float,
        d2: float,
        S: float,
        sigma: float,
        cp_flag: OptionType,
    ) -> float:
        """
        Compute's Black-Scholes theta

        Args:
            K (float): Option Strike
            r (float): Riskless Rate
            tau (float): Time to expiration in years
            d2 (float): Black-Scholes d2
            S (float): Spot Price
            sigma (float): Volatility
            cp_flag (OptionType): Option Type (put/call)

        Returns:
            float: theta
        """
        flag = -1 if cp_flag == OptionType.CALL else 1
        return -S * sp.stats.norm.pdf(d2 + sigma * np.sqrt(tau)) * sigma / (
            2 * np.sqrt(tau)
        ) - flag * (r * K * np.exp(-r * tau) * sp.stats.norm.cdf(flag * d2))

    def _european_black_scholes(
        self,
        K: float,
        r: float,
        tau: float,
        d1: float,
        d2: float,
        S: float,
        cp_flag: OptionType,
    ) -> float:
        """
        Compute's European Black-Scholes price

        Args:
            K (float): Option Strike
            r (float): Riskless Rate
            tau (float): Time to expiration in years
            d1 (float): Black-Scholes d1
            d2 (float): Black-Scholes d2
            S (float): Spot Price
            cp_flag (OptionType): Option Type (put/call)

        Returns:
            float: price
        """
        flag = 1 if cp_flag == OptionType.CALL else -1
        return -flag * np.exp(-r * tau) * K * sp.stats.norm.cdf(
            flag * d2
        ) + flag * S * sp.stats.norm.cdf(flag * d1)

    def _replication_hedge_pnl(
        self,
        gamma: np.array,
        S: np.array,
        sigma_r: np.array,
        sigma_i: np.array,
        delta_i: np.array,
        delta_r: np.array,
        mu: float,
        r: float,
        Z: np.array,
    ) -> np.array:
        """
        Compute Hedging Scheme Under Geometric Brownian Motion PnL

        Args:
            gamma (np.array): Gamma array
            S (np.array): Spot Price array
            sigma_r (np.array): Realized Volatility array
            sigma_i (np.array): Implied Volatility array
            delta_i (np.array): Implied Volatility Delta array
            delta_r (np.array): Realized Volatility Delta array
            mu (float): Stock's drift
            r (float): Riskless Rate
            Z (np.array): Random Normal(0,1) array

        Returns:
            np.array: Replication Hedge PnL
        """
        return 0.5 * gamma * np.power(S, 2) * (
            np.power(sigma_i, 2) - np.power(sigma_r, 2)
        ) * self._dt + (delta_r - delta_i) * ((mu - r) * S * self._dt + sigma_r * S * Z)

    def _compute_greeks(
        self,
        df: pd.DataFrame,
        opt: EquityOption,
        mkt: MarketDynamics,
        suffix: Literal["_r", "_i"],
    ) -> pd.DataFrame:
        """
        Compute Options Greeks For Dataframe

        Args:
            df (pd.DataFrame): DataFrame to compute Greeks on
            opt (EquityOption): Option to calculate greeks from
            mkt (MarketDynamics): Market Dynamics
            suffix (Literal['_r', '_i']): Suffix to distinguish different greeks

        Returns:
            pd.DataFrame: Dataframe with Greeks
        """
        df[f"d1{suffix}"] = self._d1(
            S=df.S, K=opt.strike, r=mkt.risk_free_rate, sigma=opt.vol, tau=df.tau
        )
        df[f"d2{suffix}"] = self._d2(d1=df[f"d1{suffix}"], sigma=opt.vol, tau=df.tau)
        df[f"delta{suffix}"] = self._delta(d1=df[f"d1{suffix}"], cp_flag=opt.type)
        df[f"gamma{suffix}"] = self._gamma(
            K=opt.strike,
            r=mkt.risk_free_rate,
            tau=df.tau,
            d2=df[f"d2{suffix}"],
            S=df.S,
            sigma=opt.vol,
        )
        df[f"theta{suffix}"] = self._theta(
            K=opt.strike,
            r=mkt.risk_free_rate,
            tau=df.tau,
            d2=df[f"d2{suffix}"],
            S=df.S,
            sigma=opt.vol,
            cp_flag=opt.type,
        )
        df[f"vega{suffix}"] = self._vega(
            K=opt.strike, r=mkt.risk_free_rate, tau=df.tau, d2=df[f"d2{suffix}"]
        )
        df[f"bs_price{suffix}"] = self._european_black_scholes(
            K=opt.strike,
            r=mkt.risk_free_rate,
            tau=df.tau,
            d1=df[f"d1{suffix}"],
            d2=df[f"d2{suffix}"],
            S=df.S,
            cp_flag=opt.type,
        )

        return df
