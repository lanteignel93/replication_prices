import pandas as pd

from dynamics import *
from instrument import *
from simulation import *
from plots import *


if __name__ == "__main__":
    # OPTION CHARACTERISTICS
    S = 100
    DAYS = 21
    SIGMA_REALIZED = 0.30
    SIGMA_IMPLIED = 0.4
    MU = 0.20

    # MARKET CHARACTERISTICS
    R = 0.04

    # JUMP CHARACTERISTICS
    JUMP_STD = 0.10
    JUMP_MEAN = 0
    JUMP_FREQ = 4

    # SIMULATION CHARACTERISTICS
    NUM_PATHS = 10
    NUM_SIMS = 10000

    # # SECTION 3.2 and 3.3 Figures
    # for K in [100, 120]:
    #     for NUM_STEPS in [21, 10000]:
    #         option_realized = EquityOption(
    #             type=OptionType.CALL,
    #             exercise=ExerciseType.EUROPEAN,
    #             spot=S,
    #             strike=K,
    #             vol=SIGMA_REALIZED,
    #             drift=MU,
    #             days_to_maturity=DAYS,
    #         )

    #         option_implied = EquityOption(
    #             type=OptionType.CALL,
    #             exercise=ExerciseType.EUROPEAN,
    #             spot=S,
    #             strike=K,
    #             vol=SIGMA_IMPLIED,
    #             drift=MU,
    #             days_to_maturity=DAYS,
    #         )

    #         market_dynamics = MarketDynamics(risk_free_rate=R)
    #         sim_parameters = SimulationParameters(
    #             num_steps=NUM_STEPS, num_paths=NUM_PATHS
    #         )

    #         sim = Simulation(parameters=sim_parameters)

    #         pnl_df = pd.DataFrame()
    #         for i in range(sim_parameters.num_paths):
    #             df = sim.run_hedge_vol_simulation(
    #                 opt_r=option_realized,
    #                 opt_i=option_implied,
    #                 market_dynamics=market_dynamics,
    #                 seed=i,
    #                 hedge_vol="realized",
    #             )
    #             pnl_df[i] = df.PnL

    #         option_price_r = df.bs_price_r.iloc[0]
    #         option_price_i = df.bs_price_i.iloc[0]

    #         plot_hedge_vol_sim(
    #             pnl_df=pnl_df,
    #             opt_r=option_realized,
    #             opt_i=option_implied,
    #             sim_parameters=sim_parameters,
    #             option_price_r=option_price_r,
    #             option_price_i=option_price_i,
    #             prefix="rv_",
    #         )

    #         pnl_df = pd.DataFrame()
    #         for i in range(sim_parameters.num_paths):
    #             df = sim.run_hedge_vol_simulation(
    #                 opt_r=option_realized,
    #                 opt_i=option_implied,
    #                 market_dynamics=market_dynamics,
    #                 seed=i,
    #                 hedge_vol="implied",
    #             )
    #             pnl_df[i] = df.PnL

    #         option_price_r = df.bs_price_r.iloc[0]
    #         option_price_i = df.bs_price_i.iloc[0]

    #         plot_hedge_vol_sim(
    #             pnl_df=pnl_df,
    #             opt_r=option_realized,
    #             opt_i=option_implied,
    #             sim_parameters=sim_parameters,
    #             option_price_r=option_price_r,
    #             option_price_i=option_price_i,
    #             prefix="iv_",
    #         )

    # # SECTION 5.1 and 5.2 Figures
    # for NUM_STEPS in [5, 50, 10000]:
    #     for K in [100, 120]:
    #         for DRIFT in [R, MU]:
    #             for hedge_vol in ["realized", "implied"]:
    #                 option_realized = EquityOption(
    #                     type=OptionType.CALL,
    #                     exercise=ExerciseType.EUROPEAN,
    #                     spot=S,
    #                     strike=K,
    #                     vol=SIGMA_REALIZED,
    #                     drift=DRIFT,
    #                     days_to_maturity=DAYS,
    #                 )

    #                 option_implied = EquityOption(
    #                     type=OptionType.CALL,
    #                     exercise=ExerciseType.EUROPEAN,
    #                     spot=S,
    #                     strike=K,
    #                     vol=SIGMA_IMPLIED,
    #                     drift=MU,
    #                     days_to_maturity=DAYS,
    #                 )

    #                 market_dynamics = MarketDynamics(risk_free_rate=R)
    #                 sim_parameters = SimulationParameters(
    #                     num_steps=NUM_STEPS, num_paths=NUM_SIMS
    #                 )

    #                 sim = Simulation(parameters=sim_parameters)
    #                 repprc_df = pd.DataFrame(
    #                     index=range(0, sim_parameters.num_paths), columns=["repprc"]
    #                 )
    #                 stock_dynamics = GeometricBrownianMotion(
    #                     option=option_realized, num_steps=NUM_STEPS, seed=seed
    #                 )
    #                 for i in range(sim_parameters.num_paths):
    #                     price = sim.run_replication_price_simulation(
    #                         opt_r=option_realized,
    #                         opt_i=option_implied,
    #                         market_dynamics=market_dynamics,
    #                         stocks_dynamics=stock_dynamics,
    #                         hedge_vol=hedge_vol,
    #                     )
    #                     repprc_df.loc[i, "repprc"] = price[0]
    #                 _, implied_price, realized_price = price

    #                 plot_replication_price_sim(
    #                     repprc_df=repprc_df,
    #                     opt_r=option_realized,
    #                     opt_i=option_implied,
    #                     sim_parameters=sim_parameters,
    #                     market_dynamics=market_dynamics,
    #                     option_price_r=realized_price,
    #                     option_price_i=implied_price,
    #                     hedge_vol=hedge_vol,
    #                 )
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

                    stock_dynamics2 = GeometricBrownianMotion(
                        parameters=stock_geo_dynamics_parameters, seed=i
                    )
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
    # STRIKE_RANGES = np.arange(85, 116, 1)
    # moneyness = np.log(STRIKE_RANGES / S)

    # for dynamics in [
    #     StockDynamicsType.JUMP_DIFFUSION_PROCESS,
    #     StockDynamicsType.GEOMETRIC_BROWNIAN_MOTION,
    # ]:
        
    #     for hedge_vol in [HedgeType.REALIZED, HedgeType.IMPLIED]:
    #         for DRIFT in [MU]:
    #             for HEDGING_FREQUENCY in [1, 5, 100]:
    #                 df = pd.DataFrame(index=moneyness)
    #                 for NUM_DAYS in [5, 10, 21, 42]:
    #                     days = NUM_DAYS
    #                     NUM_STEPS = days * HEDGING_FREQUENCY
    #                     std_mean_vec = np.zeros(len(STRIKE_RANGES))

    #                     for j, K in enumerate(STRIKE_RANGES):
    #                         if K < 100:
    #                             cp = OptionType.PUT
    #                         else:
    #                             cp = OptionType.CALL

    #                         option_realized = EquityOption(
    #                             type=cp,
    #                             exercise=ExerciseType.EUROPEAN,
    #                             spot=S,
    #                             strike=K,
    #                             vol=SIGMA_REALIZED,
    #                             drift=DRIFT,
    #                             days_to_maturity=days,
    #                         )

    #                         option_implied = EquityOption(
    #                             type=cp,
    #                             exercise=ExerciseType.EUROPEAN,
    #                             spot=S,
    #                             strike=K,
    #                             vol=SIGMA_IMPLIED,
    #                             drift=MU,
    #                             days_to_maturity=days,
    #                         )

    #                         market_dynamics = MarketDynamics(risk_free_rate=R)
    #                         sim_parameters = SimulationParameters(
    #                             num_steps=NUM_STEPS, num_paths=NUM_SIMS
    #                         )

    #                         sim = Simulation(parameters=sim_parameters)
    #                         repprc_df = pd.DataFrame(
    #                             index=range(0, sim_parameters.num_paths),
    #                             columns=["repprc"],
    #                         )

    #                         stock_jump_dynamics_parameters = (
    #                             JumpDiffusionProcessParameters(
    #                                 option=option_realized,
    #                                 num_steps=NUM_STEPS,
    #                                 lambd=JUMP_FREQ,
    #                                 m=JUMP_MEAN,
    #                                 nu=JUMP_STD,
    #                             )
    #                         )

    #                         stock_geo_dynamics_parameters = (
    #                             GeometricBrownianMotionParameters(
    #                                 option=option_realized, num_steps=NUM_STEPS
    #                             )
    #                         )

    #                         for i in range(sim_parameters.num_paths):
    #                             match dynamics:
    #                                 case StockDynamicsType.JUMP_DIFFUSION_PROCESS:
    #                                     stock_dynamics = JumpDiffusionProcess(
    #                                         parameters=stock_jump_dynamics_parameters,
    #                                         seed=i,
    #                                     )
    #                                 case StockDynamicsType.GEOMETRIC_BROWNIAN_MOTION:
    #                                     stock_dynamics = GeometricBrownianMotion(
    #                                         parameters=stock_geo_dynamics_parameters,
    #                                         seed=i,
    #                                     )
    #                                 case _:
    #                                     NotImplementedError(
    #                                         "Stock Dynamics Type not Implemented"
    #                                     )

    #                             price = sim.run_replication_price_simulation(
    #                                 opt_r=option_realized,
    #                                 opt_i=option_implied,
    #                                 market_dynamics=market_dynamics,
    #                                 stocks_dynamics=stock_dynamics,
    #                                 hedge_vol=hedge_vol,
    #                             )

    #                             repprc_df.loc[i, "repprc"] = price[0]
    #                         std_mean_vec[j] = repprc_df.std() / repprc_df.mean().abs()
    #                     df.loc[:, f"{NUM_DAYS} DTE"] = std_mean_vec

    #                 plot_repprc_dist(
    #                     df=df,
    #                     opt_r=option_realized,
    #                     opt_i=option_implied,
    #                     sim_parameters=sim_parameters,
    #                     market_dynamics=market_dynamics,
    #                     stocks_dynamics=stock_dynamics,
    #                     hedge_vol=hedge_vol,
    #                 )
    #                 print("chill")
