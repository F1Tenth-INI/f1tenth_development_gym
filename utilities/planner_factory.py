"""Factory for controller/planner instances used by CarSystem."""

import importlib


def initialize_planner(controller: str):
    if controller is None:
        planner = None
    elif controller == "mpc":
        from Control_Toolkit_ASF.Controllers.MPC.mpc_planner import mpc_planner

        planner = mpc_planner()
    elif controller == "mppi-lite-jax":
        from Control_Toolkit_ASF.Controllers.MPPILite.mppi_lite_jax_planner import MPPILitePlanner

        planner = MPPILitePlanner()
    elif controller == "rpgd-lite-jax":
        from Control_Toolkit_ASF.Controllers.MPPILite.rpgd_jax_planner import RPGDPlanner

        planner = RPGDPlanner()
    elif controller == "ftg":
        from Control_Toolkit_ASF.Controllers.FollowTheGap import ftg_planner

        importlib.reload(ftg_planner)
        planner = ftg_planner.FollowTheGapPlanner()
    elif controller == "neural":
        from Control_Toolkit_ASF.Controllers.NeuralNetImitator import nni_planner

        importlib.reload(nni_planner)
        planner = nni_planner.NeuralNetImitatorPlanner()
    elif controller == "nni-lite":
        from Control_Toolkit_ASF.Controllers.NNLite import nni_lite_planner

        importlib.reload(nni_lite_planner)
        planner = nni_lite_planner.NNLitePlanner()
    elif controller == "pp":
        from Control_Toolkit_ASF.Controllers.PurePursuit import pp_planner

        importlib.reload(pp_planner)
        planner = pp_planner.PurePursuitPlanner()
    elif controller == "stanley":
        from Control_Toolkit_ASF.Controllers.Stanley import stanley_planner

        importlib.reload(stanley_planner)
        planner = stanley_planner.StanleyPlanner()
    elif controller == "sysid":
        from Control_Toolkit_ASF.Controllers.SysId import sysid_planner

        importlib.reload(sysid_planner)
        planner = sysid_planner.SysIdPlanner()
    elif controller == "sac_agent":
        from TrainingLite.rl_racing.sac_agent_planner import RLAgentPlanner

        planner = RLAgentPlanner()
    elif controller == "manual":
        from Control_Toolkit_ASF.Controllers.Manual import manual_planner

        importlib.reload(manual_planner)
        planner = manual_planner.manual_planner()
    elif controller == "example":
        from Control_Toolkit_ASF.Controllers.ExamplePlanner import example_planner

        importlib.reload(example_planner)
        planner = example_planner.ExamplePlanner()
    elif controller == "random":
        from Control_Toolkit_ASF.Controllers.Random import random_planner

        importlib.reload(random_planner)
        planner = random_planner.random_planner()
    else:
        print(f"controller {controller} not recognized")
        raise NotImplementedError(f"{controller} is not a valid controller name for f1t")

    return planner


def if_mpc_define_cs_variables(planner):
    if hasattr(planner, "mpc"):
        horizon = planner.mpc.optimizer.mpc_horizon
        angular_control_dict = {"cs_a_{}".format(i): 0 for i in range(horizon)}
        translational_control_dict = {"cs_t_{}".format(i): 0 for i in range(horizon)}
        return angular_control_dict, translational_control_dict

    if hasattr(planner, "optimal_control_sequence"):
        horizon = len(planner.optimal_control_sequence)
        angular_control_dict = {"cs_a_{}".format(i): 0 for i in range(horizon)}
        translational_control_dict = {"cs_t_{}".format(i): 0 for i in range(horizon)}
        return angular_control_dict, translational_control_dict

    return {}, {}
