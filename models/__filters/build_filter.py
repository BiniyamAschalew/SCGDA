"""We create filter propagators as class"""
from models.__filters.bern import BernProp
from models.__filters.cheb import ChebProp
from models.__filters.mono import MonoProp


def build_filter(filter_name: str):
    # the cheb filter is by default using the normalized adjacency (instead of the official L - I)
    if filter_name == "cheb":
        return ChebProp
    elif filter_name == "bern":
        return BernProp
    elif filter_name == "mono":
        return MonoProp
    else:
        raise ValueError(f"Unknown filter: {filter_name}")