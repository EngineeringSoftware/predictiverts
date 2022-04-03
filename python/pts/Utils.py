from typing import *

import copy
import numpy as np
from scipy import stats


class Utils:

    @classmethod
    def get_option_as_boolean(cls, options, opt, default=False) -> bool:
        if opt not in options:
            return default
        else:
            # Due to limitations of CliUtils...
            return str(options.get(opt, "false")).lower() != "false"
        # end if

    @classmethod
    def get_option_as_list(cls, options, opt, default=None) -> list:
        if opt not in options:
            return copy.deepcopy(default)
        else:
            l = options[opt]
            if not isinstance(l, list):  l = [l]
            return l
        # end if

    # Summaries
    SUMMARIES_FUNCS: Dict[str, Callable[[Union[list, np.ndarray]], Union[int, float]]] = {
        "AVG": lambda l: np.mean(l) if len(l) > 0 else np.NaN,
        "SUM": lambda l: sum(l) if len(l) > 0 else np.NaN,
        "MAX": lambda l: max(l) if len(l) > 0 else np.NaN,
        "MIN": lambda l: min(l) if len(l) > 0 else np.NaN,
        "MEDIAN": lambda l: np.median(l) if len(l) > 0 and np.NaN not in l else np.NaN,
        "STDEV": lambda l: np.std(l) if len(l) > 0 else np.NaN,
        "MODE": lambda l: stats.mode(l).mode[0].item() if len(l) > 0 else np.NaN,
        "CNT": lambda l: len(l),
    }

    SUMMARIES_PRESERVE_INT: Dict[str, bool] = {
        "AVG": False,
        "SUM": True,
        "MAX": True,
        "MIN": True,
        "MEDIAN": False,
        "STDEV": False,
        "MODE": True,
        "CNT": True,
    }
