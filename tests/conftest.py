"""Pytest session configuration ensuring headless Agg matplotlib backend."""

import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
