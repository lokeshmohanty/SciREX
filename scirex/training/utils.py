# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org

"""
Module: utils.py

Utility functions for training and data processing.

Functions:
    moving_average: Compute the moving average of a sequence.

Authors:
    - Lokesh Mohanty (lokeshm@iisc.ac.in)
"""

from typing import Any

import numpy as np


def moving_average(data: Any, span: int = 1000) -> Any:
    """Compute the moving average of a sequence.

    Args:
        data: The input sequence (list, tuple, or array).
        span: The window size for the moving average.

    Returns:
        The moving average of the sequence as a numpy array.
    """
    x = np.asarray(data)
    if x.size == 0:
        return x

    N = x.size
    span = min(span, N)

    cumsum_full = np.cumsum(x, dtype=float)
    initial_indices = np.arange(1, span)
    initial_part = cumsum_full[: span - 1] / initial_indices

    diff = cumsum_full[span:] - cumsum_full[:-span]
    steady_sums = np.concatenate([cumsum_full[span - 1 : span], diff])
    steady_part = steady_sums / span

    return np.concatenate([initial_part, steady_part])
