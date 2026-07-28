"""Masked Language Model dataset implementation for PyDTNN."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from collections.abc import Generator

import numpy as np
from spacy.language import Language

from pydtnn.datasets.abstract import Dataset

__all__ = ("MaskLang",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model

TRAIN_NSAMPLES = 10000
TEST_NSAMPLES = 2000
INPUT_SHAPE = (1, 1, 1)
OUTPUT_SHAPE = (1,)


class MaskLang(Dataset):
    """
    Masked Language Model Dataset

    NOTE: Source unclear
    TODO: Load original dataset

    Source (SHA1): ???

    Normalize (z-score):
    offset: ???
    scale:  ???
    """

    def __init__(
        self,
        model: Model,
        preprocess: int = 0,
        embedl: int = 512,
        max_sentence: int = 512,
        split_token: str = "<translation>",
        force_test_as_validation: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Initialize the MaskLang dataset.

        Args:
            model: The model instance.
            preprocess: Number of samples to pre-process.
            embedl: Embedding length.
            max_sentence: Maximum sentence length.
            split_token: Token used for splitting.
            force_test_as_validation: Whether to force test set as validation.
            debug: Debug mode flag.
        """

        self.model = model
        self.num_preprocess = preprocess
        self.split_token = split_token
        self.max_sentence = max_sentence
        self.embedl = embedl
        self.train_path = self.model.dataset_path
        self.test_path = self.model.dataset_path
        self.test_as_validation = force_test_as_validation
        self.dtype = self.model.dtype
        self.lang = self.model.dataset_lang

        if self.num_preprocess > 0:
            self._data_generator = self._actual_data_generator_preprocess
        else:
            self._data_generator = self._actual_data_generator_normal

        super().__init__(
            model,
            TRAIN_NSAMPLES,
            TEST_NSAMPLES,
            INPUT_SHAPE,
            OUTPUT_SHAPE,
            force_test_as_validation=force_test_as_validation,
            debug=debug,
        )

    def _model_init(self) -> None:
        """Initialize actual data by loading, partitioning, and optionally preprocessing."""
        # Actual
        self.load_data()
        self.make_train_val_partitions()
        if self.num_preprocess > 0:
            self.preprocess(size=self.num_preprocess)
            self.make_train_val_partitions()

        # Synthetic
        # self.train_val_nsamples = 10000
        # self.train_nsamples = -1
        # self.make_train_val_partitions()
        # self.src_embeddings = random.random(
        #     (self.train_val_nsamples, 1, self.max_sentence, self.embedl)
        # ).astype(dtype=self.dtype)
        # self.tgt_embeddings = random.random(
        #     (self.train_val_nsamples, 1, self.max_sentence, self.embedl)
        # ).astype(dtype=self.dtype)

    def load_data(self) -> None:
        """Load raw text data from the dataset path."""
        self.dictionary = self.get_dictionary(self.lang)
        self.mask = self.dictionary("Mask")[0]
        self.mask = np.zeros(self.mask.vector.shape, dtype=self.dtype)
        file = open(self.train_path, "r")
        self.lines = file.readlines()
        file.close()
        self.lines = [line.replace("\n", "") for line in self.lines]

        self.train_val_nsamples = len(self.lines)
        logger.info(self.train_val_nsamples)
        self.train_nsamples = -1

    def get_dictionary(self, language: str) -> Language:
        """
        Load a spaCy language model.

        Args:
            language: Language code.

        Returns:
            The loaded spaCy model.
        """
        import spacy

        table = {"en": "en_core_web_md", "de": "de_core_news_md"}
        if language in table:
            language = table[language]
        return spacy.load(language)

    def make_train_val_partitions(self) -> None:
        """Create training and validation partitions based on model configuration."""
        val_split = self.model.validation_split
        if self.train_nsamples < 0:
            s = np.arange(self.train_val_nsamples)
            if self.model.augment_shuffle:
                self.model.random.shuffle(s)
            self.train_nsamples = int(self.train_val_nsamples * (1 - val_split) // 1)
            self.train_indices = s[: self.train_nsamples]
            self.val_indices = s[self.train_nsamples:]
            self.val_nsamples = len(self.val_indices)
            self.test_nsamples = self.val_nsamples

    def _actual_data_generator_normal(
        self, part: Dataset.Part
    ) -> Generator[tuple[np.ndarray, np.ndarray], Any, None]:
        """
        Generator for on-the-fly data processing.

        Args:
            part: The partition to generate data for.
        """
        batch_size = self.model.batch_size
        rank = self.model.rank

        for i in range(self.train_val_nsamples // batch_size):
            window = (i * batch_size + rank * batch_size, i * batch_size + (rank + 1) * batch_size)
            src_embeddings = np.zeros(
                (batch_size, 1, self.max_sentence, self.embedl), dtype=self.dtype
            )
            tgt_embeddings = np.zeros(
                (batch_size, 1, self.max_sentence, self.embedl), dtype=self.dtype
            )
            for i, doc in enumerate(self.dictionary.pipe(self.lines[window[0]: window[1]])):
                mask = self.model.random.integers(0, len(doc))
                for j, word in enumerate(doc):
                    if j > self.max_sentence:
                        break
                    if j == mask:
                        src_embeddings[i, 0, j] = self.mask
                    else:
                        src_embeddings[i, 0, j] = word.vector
                    tgt_embeddings[i, 0, j] = word.vector
            x = src_embeddings
            y = tgt_embeddings
            yield x, y

    def _actual_data_generator_preprocess(
        self, part: Dataset.Part
    ) -> Generator[tuple[np.ndarray, np.ndarray], Any, None]:
        """
        Generator for pre-processed data.

        Args:
            part: The partition to generate data for.
        """
        batch_size = self.model.batch_size
        rank = self.model.rank

        for i in range(self.train_val_nsamples // batch_size):
            window = (i * batch_size + rank * batch_size, i * batch_size + (rank + 1) * batch_size)
            x = self.src_embeddings[window[0]: window[1]]
            y = self.tgt_embeddings[window[0]: window[1]]
            yield x, y

    def _synthetic_data_generator(self) -> Generator[tuple[np.ndarray, np.ndarray], Any, None]:
        """Generator for synthetic data."""
        batch_size = self.model.batch_size
        rank = self.model.rank

        for i in range(self.train_val_nsamples // batch_size):
            window = (i * batch_size + rank * batch_size, i * batch_size + (rank + 1) * batch_size)
            x = self.src_embeddings[window[0]: window[1]]
            y = self.tgt_embeddings[window[0]: window[1]]
            yield x, y

    # === Preprocess ===
    def preprocess(self, size: int | None = None) -> None:
        """
        Pre-process text data into embeddings.

        Args:
            size: Number of samples to process.
        """
        if size is None:
            size = len(self.lines)
        self.train_val_nsamples = size
        self.src_embeddings = np.zeros((size, 1, self.max_sentence, self.embedl), dtype=self.dtype)
        self.tgt_embeddings = np.zeros((size, 1, self.max_sentence, self.embedl), dtype=self.dtype)
        for i, doc in enumerate(self.dictionary.pipe(self.lines[0:size])):
            mask = self.model.random.integers(0, len(doc))
            # self.src_embeddings[i,0,0:len(doc)] = doc
            # self.tgt_embeddings[i,0,]
            # self.src_embeddings[i,0,mask] = self.mask
            for j, word in enumerate(doc):
                if j > self.max_sentence:
                    break
                if j == mask:
                    self.src_embeddings[i, 0, j] = self.mask
                else:
                    self.src_embeddings[i, 0, j] = word.vector

    # === Preprocess ===
