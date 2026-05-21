# Lazy imports so kitchen (gymnasium-only) does not require mj_envs.


def __getattr__(name):
    if name == "AdroitEnv":
        from .adroit import AdroitEnv
        return AdroitEnv
    if name == "MetaWorldEnv":
        from .metaworld import MetaWorldEnv
        return MetaWorldEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AdroitEnv", "MetaWorldEnv"]
