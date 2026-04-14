"""
Compatibility shims for pickled DreamDiffusion checkpoints.

DreamDiffusion checkpoints can contain instances of config classes such as
`config.Config_Generative_Model`. When we load those pickles outside the
original repository layout, PyTorch tries to import the `config` module and
resolve those class names. This package already exists in this repo as a
directory for configuration files, so we provide lightweight placeholder
classes here to satisfy unpickling.

The checkpoint loader only needs these objects to exist long enough to unwrap
the state_dict; none of the original DreamDiffusion config behavior is needed.
"""

from __future__ import annotations


class DreamDiffusionConfigStub:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _make_stub_class(name: str):
    stub = type(name, (DreamDiffusionConfigStub,), {})
    globals()[name] = stub
    return stub


def __getattr__(name: str):
    if name.startswith("Config"):
        return _make_stub_class(name)
    raise AttributeError(name)


Config_Generative_Model = _make_stub_class("Config_Generative_Model")
