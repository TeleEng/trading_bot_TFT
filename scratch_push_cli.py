import sys
import builtins
_orig_open = builtins.open
builtins.open = lambda file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None: _orig_open(file, mode, buffering, 'utf-8' if 'w' in mode and 'b' not in mode and encoding is None else encoding, errors, newline, closefd, opener)

from kaggle.cli import main

sys.argv = ['kaggle', 'kernels', 'push', '-p', 'd:/Work/trading_bot/kaggle_pull']
main()
