# coding=utf-8
import types
from typing import Any, Optional

import omegaconf

"""
Note: 'trajectory_container_tools' import statements is required so that function 
'dictconfig_to_class_resolved_dict()' has access to its namespace by default. 
"""
import trajectory_container_tools as tct


def is_cfg_key_exist(
    cfg: omegaconf.DictConfig,
    key: str,
    raise_error: bool = False,
    error_msg_prepend_key_parent: Optional[str] = None,
) -> bool:
    # (NICE TO HAVE) ToDo: implement test case
    """
    Checks if `key: value` exist or key is missing (i.e. `key: ???`) in the configuration dict.
    Support dot separated keys (e.g., key: "one_dim_transition_model.normalizer_type")

    :param cfg: a dictionary config.
    :param key: The key to check for in the configuration.
    :param error_msg_prepend_key_parent:
    :param raise_error: Raise Exception instead of returning False on faillure.
    :return: True if the key exist.
    :raise KeyError: Raise KeyError if key does not exist.
    :raise omegaconf.errors.MissingMandatoryValue: Raise MissingMandatoryValue if is `key: ???`.
    """
    key_parent = (
        f"{error_msg_prepend_key_parent}." if error_msg_prepend_key_parent else ""
    )
    error_msg = f"Missing cfg key `{key_parent}{key}`"
    status = omegaconf.OmegaConf.select(
        cfg, key, default=KeyError(error_msg), throw_on_missing=raise_error
    )
    if isinstance(status, KeyError):
        if raise_error:
            raise status
        return False
    else:
        return True


def dictconfig_to_class_resolved_dict(
    cfg: omegaconf.DictConfig, namespaces: dict = {"tct": tct}
) -> dict | list | None | str | Any:
    """
    Converts a given omegaconf.DictConfig object into a dictionary with the values resolved as
    class attributes from the global namespace.

    Note: Support both 'trajectory_container_tools' namespace and 'tct' acronym namespace

    Iterates through items in the config and evaluates string references to nested attributes in
    the global namespace. Supports nested lookups for attribute paths.

    Usage:

    >>> import omegaconf
    >>> import trajectory_container_tools as tct
    >>>
    >>> dict_cfg = omegaconf.OmegaConf.create(
    >>>     {
    >>>         "/odom": "tct.dataclasses.NavMsgsOdometry",
    >>>         "/tf": "tct.dataclasses.Tf2MsgsTFMessage",
    >>>     }
    >>> )
    >>>
    >>> expected_cfg_output = {
    >>>     "/odom": tct.dataclasses.NavMsgsOdometry,
    >>>     "/tf": tct.dataclasses.Tf2MsgsTFMessage,
    >>> }
    >>>
    >>> assert dictconfig_to_class_resolved_dict(dict_cfg) == expected_cfg_output

    :param cfg: The configuration object to be converted. Must be an instance of omegaconf.DictConfig.
    :param namespaces:
    :return: A Python dictionary with resolved class attributes or values. Returns None, str, or other types based on the data in the provided config.
    """
    _ = tct.version

    feature_config = omegaconf.OmegaConf.to_object(cfg)
    v_namespace: types.ModuleType | None = None

    for key, value in namespaces.items():
        globals()[key] = value

    for k, v in feature_config.items():
        v = v.split(".")
        for idx, each in enumerate(v):
            if idx == 0:
                v_namespace = globals().get(each)
            else:
                v_namespace = v_namespace.__getattribute__(each)
        feature_config[k] = v_namespace
    return feature_config
