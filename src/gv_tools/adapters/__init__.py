"""Built-in instrument adapters."""

from .base import InstrumentAdapter
from .lufft_ws800 import LufftWS800Adapter, RawLufftWS800Product
from .metone_aio import MetOneAIOAdapter, RawMetOneAIOProduct
from .parsivel import ParsivelAdapter, ParsivelSourcePlatform, RawParsivelProduct
from .rm_young_aio import RMYoungAIOAdapter, RawRMYoungAIOProduct

__all__ = [
    "InstrumentAdapter", "LufftWS800Adapter", "RawLufftWS800Product", "MetOneAIOAdapter", "RawMetOneAIOProduct", "ParsivelAdapter", "ParsivelSourcePlatform",
    "RawParsivelProduct", "RMYoungAIOAdapter", "RawRMYoungAIOProduct",
]
