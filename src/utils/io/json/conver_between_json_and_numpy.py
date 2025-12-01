from typing import Dict

import numpy as np


def convert_numpy_objects_to_str(dict_to_convert: Dict) -> Dict:
    new = {}
    for k, v in dict_to_convert.items():
        if isinstance(v, dict):
            new[k] = convert_numpy_objects_to_str(v)
        elif isinstance(v, (list, tuple, set)):
            new[k] = [convert_numpy_objects_to_str(e) for e in v if isinstance(e, dict)]
        else:
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                new[k] = str(v)
            else:
                new[k] = v
    return new


def convert_str_to_numpy_objects(dict_to_convert: Dict) -> Dict:
    new = {}
    for k, v in dict_to_convert.items():
        if isinstance(v, dict):
            new[k] = convert_str_to_numpy_objects(v)
        elif isinstance(v, (list, tuple, set)):
            for e in v:
                if isinstance(e, dict):
                    new[k] = [
                        convert_str_to_numpy_objects(e)
                        for e in v
                        if isinstance(e, dict)
                    ]
                else:
                    new[k] = v
        else:
            if v == "nan":
                new[k] = np.nan
            elif v == "inf":
                new[k] = np.inf
            elif v == "-inf":
                new[k] = -np.inf
            else:
                new[k] = v
    return new
