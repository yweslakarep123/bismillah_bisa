# Lazy imports so kitchen (gymnasium-only) does not require pytorch3d/mujoco.


def __getattr__(name):
    if name == "MujocoPointcloudWrapperAdroit":
        from flow_policy_3d.gym_util.mjpc_wrapper import MujocoPointcloudWrapperAdroit
        return MujocoPointcloudWrapperAdroit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MujocoPointcloudWrapperAdroit"]
