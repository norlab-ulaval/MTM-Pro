# coding=utf-8
import time

import numpy as np
import torch
import omegaconf
from typing import Optional, Tuple, Union

from tools.console_tools.format import ConsoleFormat
from tools.console_tools.message import consol_msg_motion_model_learner_one_line


def seed_me(
    cfg: omegaconf.DictConfig, output_torch_rdn_generator: bool = False
) -> Union[Optional[int], Tuple[Optional[int], torch.Generator]]:
    SEED = cfg.seed
    np.random.seed(SEED)
    generator = torch.Generator(device=cfg.device)
    if SEED:
        torch.manual_seed(SEED)
        generator.manual_seed(SEED)
    if output_torch_rdn_generator:
        return SEED, generator
    else:
        return SEED


def wallclock_timer(start_time: float, max_second: float, timer_on: bool = True) -> bool:
    if timer_on:
        current_time = time.time()
        delta_time = current_time - start_time
        still_have_time = delta_time < max_second
        if not still_have_time:
            consol_msg_motion_model_learner_one_line(
                f"{ConsoleFormat.MSG_WARNING_FORMAT}Waited for {max_second}, trajectory "
                "rollout is probaly "
                f"hanging. Exiting feedback loop{ConsoleFormat.MSG_END_FORMAT}"
            )
    else:
        still_have_time = True

    return still_have_time
