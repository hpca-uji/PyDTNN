"""Fused layers test group"""

import logging
logger = logging.getLogger(__name__)

from pydtnn.tests.conv_2d_relu import Conv2DReluTestCase
from pydtnn.tests.conv_2d_batch_normalization import Conv2DBatchNormalizationTestCase
from pydtnn.tests.conv_2d_batch_normalization_relu import Conv2DBatchNormalizationReluTestCase
from pydtnn.tests.batch_normalization_relu import BatchNormalizationReluTestCase
