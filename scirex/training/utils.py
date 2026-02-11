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

    # Use cumsum for efficient sliding window
    # Convert to float for division

    cumsum_full = np.cumsum(x, dtype=float)

    # Part 1: Initial ramping up average
    # indices 0 to span-2 (length span-1)
    # average[i] = cumsum[i] / (i+1)
    initial_indices = np.arange(1, span)
    initial_part = cumsum_full[: span - 1] / initial_indices

    # Part 2: Steady state average
    # steady_sum[i] = cumsum[i + span - 1] - cumsum[i - 1]
    # But wait, original code was: out[span:] = out[span:] - out[:-span]
    # out[k] (where k >= span) becomes cumsum[k] - cumsum[k-span]
    # This corresponds to sum of x[k-span+1 ... k]
    # Then it takes out[span-1:] / span.
    # out[span-1] is just cumsum[span-1] (sum of first span elements)
    # out[span] is cumsum[span] - cumsum[0] (sum of x[1]...x[span])

    # So we need:
    # val_at_span_minus_1 = cumsum[span-1]
    # val_at_span = cumsum[span] - cumsum[0]
    # ...
    # val_at_last = cumsum[-1] - cumsum[-1-span]

    # Let's follow the numpy logic exactly but functionally.
    # out[span:] - out[:-span]
    # original 'out' is cumsum.
    # let A = cumsum[span:]
    # let B = cumsum[:-span]
    # diff = A - B
    # The 'steady' part of the sum (starting from index span) is diff.
    # The element at 'span-1' is just cumsum[span-1].

    # So the full sequence of sums for the window of size 'span' is:
    # [cumsum[span-1],  diff[0], diff[1], ...]

    diff = cumsum_full[span:] - cumsum_full[:-span]
    steady_sums = np.concatenate([cumsum_full[span - 1 : span], diff])

    steady_part = steady_sums / span

    return np.concatenate([initial_part, steady_part])
