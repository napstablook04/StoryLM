import importlib.metadata
from .tokenizer import train_bpe

try:
    __version__ = importlib.metadata.version("storylm")
except importlib.metadata.PackageNotFoundError:
    pass
