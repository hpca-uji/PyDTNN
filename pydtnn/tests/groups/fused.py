"""Fused layers test group"""

import logging

from pydtnn.tests.batch_normalization_relu import BatchNormalizationReluTestCase
from pydtnn.tests.conv_2d_batch_normalization import Conv2DBatchNormalizationTestCase
from pydtnn.tests.conv_2d_batch_normalization_relu import Conv2DBatchNormalizationReluTestCase
from pydtnn.tests.conv_2d_relu import Conv2DReluTestCase

logger = logging.getLogger(__name__)
