import pandas as pd
import numpy as np

from dynamics import (
    MarketDynamics,
    StockDynamicsType,
    GeometricBrownianMotion,
    GeometricBrownianMotionParameters,
    JumpDiffusionProcess,
    JumpDiffusionProcessParameters,
)
from instrument import (
    OptionType,
    ExerciseType,
    HedgeType,
    EquityOption,
)
from simulation import (
    Simulation,
    SimulationParameters,
)

from plots import (
    plot_repprc_dist,
    plot_hedge_vol_sim,
    plot_replication_price_sim,
)

# OPTION CHARACTERISTICS
S = 100
DAYS = 21
SIGMA_REALIZED = 0.36
SIGMA_IMPLIED = 0.4
MU = 0.08

# MARKET CHARACTERISTICS
R = 0.04

# JUMP CHARACTERISTICS
JUMP_STD = 0.10
JUMP_MEAN = 0
JUMP_FREQ = 4

# SIMULATION CHARACTERISTICS
NUM_PATHS = 10
NUM_SIMS = 5000

SEED = 100

def main():
    put_min_strike_start = S * 0.5
    call_max_strike_start = S * 1.5
    starting_strike = S

    for optimized_ratio in [1]:
        for hedging_frequency in [1, 2]:
            valid_moneyness = pd.DataFrame(
                index=np.arange(2, 51, 1), columns=["Call", "Put"]
            )

            ratio_error_tolerance = 0.05
            for DTE in np.arange(2, 51, 1):
                for OPTION_TYPE in [OptionType.CALL, OptionType.PUT]:
                    market_dynamics = MarketDynamics(risk_free_rate=R)
                    NUM_STEPS = DTE * hedging_frequency
                    ratio_error = 1
                    noise_signal_ratio = 10
                    match OPTION_TYPE:
                        case OptionType.CALL:
                            curr_strike = (call_max_strike_start + starting_strike) / 2
                        case OptionType.PUT:
                            curr_strike = (put_min_strike_start + starting_strike) / 2
                    count = 0
                    while abs(ratio_error) > abs(ratio_error_tolerance):
                        count += 1
                        moneyness = np.log(
                            curr_strike
                            / (S * np.exp(market_dynamics.risk_free_rate * DTE / 252))
                        )
                        option_realized = EquityOption(
                            type=OPTION_TYPE,
                            exercise=ExerciseType.EUROPEAN,
                            spot=S,
                            strike=curr_strike,
                            vol=SIGMA_REALIZED,
                            drift=MU,
                            days_to_maturity=DTE,
                        )
                        option_implied = EquityOption(
                            type=OPTION_TYPE,
                            exercise=ExerciseType.EUROPEAN,
                            spot=S,
                            strike=curr_strike,
                            vol=SIGMA_IMPLIED,
                            drift=MU,
                            days_to_maturity=DTE,
                        )
                        sim_parameters = SimulationParameters(
                            num_steps=NUM_STEPS, num_paths=NUM_SIMS
                        )
                        sim = Simulation(parameters=sim_parameters)

                        stock_geo_dynamics_parameters = (
                            GeometricBrownianMotionParameters(
                                option=option_realized, num_steps=NUM_STEPS
                            )
                        )

                        repprc_df = pd.DataFrame(
                            index=range(0, sim_parameters.num_paths),
                            columns=["repprc"],
                        )

                        for i in range(sim_parameters.num_paths):
                            stock_dynamics = GeometricBrownianMotion(
                                parameters=stock_geo_dynamics_parameters,
                            )
                            price = sim.run_replication_price_simulation(
                                opt_r=option_realized,
                                opt_i=option_implied,
                                market_dynamics=market_dynamics,
                                stocks_dynamics=stock_dynamics,
                                hedge_vol=HedgeType.IMPLIED,
                            )

                            repprc_df.loc[i, "repprc"] = price[0]
                        noise_signal_ratio = (
                            repprc_df.std() / repprc_df.mean().abs()
                        ).values[0]
                        ratio_error = noise_signal_ratio - optimized_ratio

                        match OPTION_TYPE:
                            case OptionType.CALL:
                                if noise_signal_ratio > optimized_ratio:
                                    call_max_strike = curr_strike
                                else:
                                    starting_strike = curr_strike

                                curr_strike = round(
                                    (call_max_strike + starting_strike) / 2, 1
                                )

                            case OptionType.PUT:
                                if noise_signal_ratio > optimized_ratio:
                                    put_min_strike = curr_strike
                                else:
                                    starting_strike = curr_strike

                                curr_strike = round(
                                    (put_min_strike + starting_strike) / 2, 1
                                )

                        if (
                            abs(ratio_error) < abs(ratio_error_tolerance)
                            or abs(curr_strike - S) < 0.5
                            or count > 25
                        ):
                            print(
                                DTE,
                                curr_strike,
                                moneyness,
                                ratio_error,
                                optimized_ratio,
                                noise_signal_ratio,
                                hedging_frequency,
                            )
                            if abs(curr_strike - S) < 0.5 or count > 25:
                                moneyness = 0
                                ratio_error = 0

                        valid_moneyness.loc[
                            DTE, OPTION_TYPE.name.capitalize()
                        ] = moneyness

            valid_moneyness.to_csv(
                f"/home/laurent/latex_proof/python/data/{optimized_ratio:.1f}_ratio_{hedging_frequency}_hf_valid_moneyness.csv"
            )

    # SECTION 3.2 and 3.3 Figures
    for K in [100, 120]:
        for NUM_STEPS in [21, 10000]:
            option_realized = EquityOption(
                type=OptionType.CALL,
                exercise=ExerciseType.EUROPEAN,
                spot=S,
                strike=K,
                vol=SIGMA_REALIZED,
                drift=MU,
                days_to_maturity=DAYS,
            )

            option_implied = EquityOption(
                type=OptionType.CALL,
                exercise=ExerciseType.EUROPEAN,
                spot=S,
                strike=K,
                vol=SIGMA_IMPLIED,
                drift=MU,
                days_to_maturity=DAYS,
            )

            market_dynamics = MarketDynamics(risk_free_rate=R)
            sim_parameters = SimulationParameters(
                num_steps=NUM_STEPS, num_paths=NUM_PATHS
            )

            sim = Simulation(parameters=sim_parameters)

            pnl_df = pd.DataFrame()
            for i in range(sim_parameters.num_paths):
                df = sim.run_hedge_vol_simulation(
                    opt_r=option_realized,
                    opt_i=option_implied,
                    market_dynamics=market_dynamics,
                    seed=i,
                    hedge_vol="realized",
                )
                pnl_df[i] = df.PnL

            option_price_r = df.bs_price_r.iloc[0]
            option_price_i = df.bs_price_i.iloc[0]

            plot_hedge_vol_sim(
                pnl_df=pnl_df,
                opt_r=option_realized,
                opt_i=option_implied,
                sim_parameters=sim_parameters,
                option_price_r=option_price_r,
                option_price_i=option_price_i,
                prefix="rv_",
            )

            pnl_df = pd.DataFrame()
            for i in range(sim_parameters.num_paths):
                df = sim.run_hedge_vol_simulation(
                    opt_r=option_realized,
                    opt_i=option_implied,
                    market_dynamics=market_dynamics,
                    seed=i,
                    hedge_vol="implied",
                )
                pnl_df[i] = df.PnL

            option_price_r = df.bs_price_r.iloc[0]
            option_price_i = df.bs_price_i.iloc[0]

            plot_hedge_vol_sim(
                pnl_df=pnl_df,
                opt_r=option_realized,
                opt_i=option_implied,
                sim_parameters=sim_parameters,
                option_price_r=option_price_r,
                option_price_i=option_price_i,
                prefix="iv_",
            )

    # SECTION 5.1 and 5.2 Figures
    for NUM_STEPS in [5, 50, 10000]:
            for K in [100, 120]:
                for DRIFT in [R, MU]:
                    for hedge_vol in ["realized", "implied"]:
                        option_realized = EquityOption(
                            type=OptionType.CALL,
                            exercise=ExerciseType.EUROPEAN,
                            spot=S,
                            strike=K,
                            vol=SIGMA_REALIZED,
                            drift=DRIFT,
                            days_to_maturity=DAYS,
                        )

                        option_implied = EquityOption(
                            type=OptionType.CALL,
                            exercise=ExerciseType.EUROPEAN,
                            spot=S,
                            strike=K,
                            vol=SIGMA_IMPLIED,
                            drift=MU,
                            days_to_maturity=DAYS,
                        )

                        market_dynamics = MarketDynamics(risk_free_rate=R)
                        sim_parameters = SimulationParameters(
                            num_steps=NUM_STEPS, num_paths=NUM_SIMS
                        )

                        sim = Simulation(parameters=sim_parameters)
                        repprc_df = pd.DataFrame(
                            index=range(0, sim_parameters.num_paths), columns=["repprc"]
                        )
                        stock_dynamics = GeometricBrownianMotion(
                            option=option_realized, num_steps=NUM_STEPS, seed=SEED
                        )
                        for i in range(sim_parameters.num_paths):
                            price = sim.run_replication_price_simulation(
                                opt_r=option_realized,
                                opt_i=option_implied,
                                market_dynamics=market_dynamics,
                                stocks_dynamics=stock_dynamics,
                                hedge_vol=hedge_vol,
                            )
                            repprc_df.loc[i, "repprc"] = price[0]
                        _, implied_price, realized_price = price

                        plot_replication_price_sim(
                            repprc_df=repprc_df,
                            opt_r=option_realized,
                            opt_i=option_implied,
                            sim_parameters=sim_parameters,
                            market_dynamics=market_dynamics,
                            option_price_r=realized_price,
                            option_price_i=implied_price,
                            hedge_vol=hedge_vol,
                        )
    # SECTION 6.3
    for NUM_STEPS in [5, 10, 21, 42, 1000]:
        for K in [100, 120]:
            for DRIFT in [R, MU]:
                for hedge_vol in [HedgeType.REALIZED, HedgeType.IMPLIED]:
                    if NUM_STEPS <= 42:
                        days = NUM_STEPS
                    else:
                        days = 50
                    option_realized = EquityOption(
                        type=OptionType.CALL,
                        exercise=ExerciseType.EUROPEAN,
                        spot=S,
                        strike=K,
                        vol=SIGMA_REALIZED,
                        drift=DRIFT,
                        days_to_maturity=days,
                    )

                    option_implied = EquityOption(
                        type=OptionType.CALL,
                        exercise=ExerciseType.EUROPEAN,
                        spot=S,
                        strike=K,
                        vol=SIGMA_IMPLIED,
                        drift=MU,
                        days_to_maturity=days,
                    )

                    market_dynamics = MarketDynamics(risk_free_rate=R)
                    sim_parameters = SimulationParameters(
                        num_steps=NUM_STEPS, num_paths=NUM_SIMS
                    )

                    sim = Simulation(parameters=sim_parameters)
                    repprc_df = pd.DataFrame(
                        index=range(0, sim_parameters.num_paths), columns=["repprc"]
                    )

                    stock_jump_dynamics_parameters = (
                                JumpDiffusionProcessParameters(
                                    option=option_realized,
                                    num_steps=NUM_STEPS,
                                    lambd=JUMP_FREQ,
                                    m=JUMP_MEAN,
                                    nu=JUMP_STD,
                                )
                            )

                    stock_geo_dynamics_parameters = (
                        GeometricBrownianMotionParameters(
                            option=option_realized, num_steps=NUM_STEPS
                        )
                    )
                    for i in range(sim_parameters.num_paths):
                        stock_dynamics = JumpDiffusionProcess(
                            parameters=stock_jump_dynamics_parameters, seed=i
                        )
                        price = sim.run_replication_price_simulation(
                            opt_r=option_realized,
                            opt_i=option_implied,
                            market_dynamics=market_dynamics,
                            stocks_dynamics=stock_dynamics,
                            hedge_vol=hedge_vol,
                        )

                        repprc_df.loc[i, "repprc"] = price[0]

                
                    price = sim.run_replication_price_simulation(
                        opt_r=option_realized,
                        opt_i=option_implied,
                        market_dynamics=market_dynamics,
                        stocks_dynamics=stock_dynamics,
                        hedge_vol=hedge_vol,
                    )
                    _, implied_price, realized_price = price

                    plot_replication_price_sim(
                        repprc_df=repprc_df,
                        opt_r=option_realized,
                        opt_i=option_implied,
                        sim_parameters=sim_parameters,
                        market_dynamics=market_dynamics,
                        stocks_dynamics=stock_dynamics,
                        option_price_r=realized_price,
                        option_price_i=implied_price,
                        hedge_vol=hedge_vol,
                    )

        # SECTION 6.4 - 6.5
        STRIKE_RANGES = np.arange(85, 116, 1)
        moneyness = np.log(STRIKE_RANGES / S)

        for dynamics in [
        StockDynamicsType.JUMP_DIFFUSION_PROCESS,
        StockDynamicsType.GEOMETRIC_BROWNIAN_MOTION,
        ]:

            for hedge_vol in [HedgeType.REALIZED, HedgeType.IMPLIED]:
                for DRIFT in [MU]:
                    for HEDGING_FREQUENCY in [1, 5, 100]:
                        df = pd.DataFrame(index=moneyness)
                        for NUM_DAYS in [5, 10, 21, 42]:
                            days = NUM_DAYS
                            NUM_STEPS = days * HEDGING_FREQUENCY
                            std_mean_vec = np.zeros(len(STRIKE_RANGES))

                            for j, K in enumerate(STRIKE_RANGES):
                                if K < 100:
                                    cp = OptionType.PUT
                                else:
                                    cp = OptionType.CALL

                                option_realized = EquityOption(
                                    type=cp,
                                    exercise=ExerciseType.EUROPEAN,
                                    spot=S,
                                    strike=K,
                                    vol=SIGMA_REALIZED,
                                    drift=DRIFT,
                                    days_to_maturity=days,
                                )

                                option_implied = EquityOption(
                                    type=cp,
                                    exercise=ExerciseType.EUROPEAN,
                                    spot=S,
                                    strike=K,
                                    vol=SIGMA_IMPLIED,
                                    drift=MU,
                                    days_to_maturity=days,
                                )

                                market_dynamics = MarketDynamics(risk_free_rate=R)
                                sim_parameters = SimulationParameters(
                                    num_steps=NUM_STEPS, num_paths=NUM_SIMS
                                )

                                sim = Simulation(parameters=sim_parameters)
                                repprc_df = pd.DataFrame(
                                    index=range(0, sim_parameters.num_paths),
                                    columns=["repprc"],
                                )

                                stock_jump_dynamics_parameters = (
                                    JumpDiffusionProcessParameters(
                                        option=option_realized,
                                        num_steps=NUM_STEPS,
                                        lambd=JUMP_FREQ,
                                        m=JUMP_MEAN,
                                        nu=JUMP_STD,
                                    )
                                )

                                stock_geo_dynamics_parameters = (
                                    GeometricBrownianMotionParameters(
                                        option=option_realized, num_steps=NUM_STEPS
                                    )
                                )

                                for i in range(sim_parameters.num_paths):
                                    match dynamics:
                                        case StockDynamicsType.JUMP_DIFFUSION_PROCESS:
                                            stock_dynamics = JumpDiffusionProcess(
                                                parameters=stock_jump_dynamics_parameters,
                                                seed=i,
                                            )
                                        case StockDynamicsType.GEOMETRIC_BROWNIAN_MOTION:
                                            stock_dynamics = GeometricBrownianMotion(
                                                parameters=stock_geo_dynamics_parameters,
                                                seed=i,
                                            )
                                        case _:
                                            NotImplementedError(
                                                "Stock Dynamics Type not Implemented"
                                            )

                                    price = sim.run_replication_price_simulation(
                                        opt_r=option_realized,
                                        opt_i=option_implied,
                                        market_dynamics=market_dynamics,
                                        stocks_dynamics=stock_dynamics,
                                        hedge_vol=hedge_vol,
                                    )

                                    repprc_df.loc[i, "repprc"] = price[0]
                                std_mean_vec[j] = repprc_df.std() / repprc_df.mean().abs()
                            df.loc[:, f"{NUM_DAYS} DTE"] = std_mean_vec

                        plot_repprc_dist(
                            df=df,
                            opt_r=option_realized,
                            opt_i=option_implied,
                            sim_parameters=sim_parameters,
                            market_dynamics=market_dynamics,
                            stocks_dynamics=stock_dynamics,
                            hedge_vol=hedge_vol,
                        )
                        

if __name__ == "__main__":
    main()