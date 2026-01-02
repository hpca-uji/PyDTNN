from typing import TYPE_CHECKING

import numpy as np

from pydtnn.datasets.dataset import Dataset
from pydtnn.utils import random


if TYPE_CHECKING:
    from pydtnn.model import Model

TRAIN_NSAMPLES = 10000
TEST_NSAMPLES = 2000
INPUT_SHAPE = (1,)
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

    def __init__(self, model: "Model", preprocess=True, embedl=512, max_sentence=512, split_token="<translation>", force_test_as_validation=False, debug=False):
        super().__init__(model, TRAIN_NSAMPLES, TEST_NSAMPLES, INPUT_SHAPE, OUTPUT_SHAPE, force_test_as_validation=force_test_as_validation, debug=debug)

        self.do_preprocess = preprocess
        self.split_token = split_token
        self.max_sentence = max_sentence
        self.embedl = embedl
        self.train_path = self.model.dataset_path
        self.test_path = self.model.dataset_path
        self.test_as_validation = force_test_as_validation
        self.dtype = self.model.dtype
        self.model = model
        self.lang = self.model.dataset_lang

    def _init_actual_data(self):
        # Actual
        self.load_data()
        self.make_train_val_partitions()
        if not self.do_preprocess:
            self._actual_data_generator = self._actual_data_generator_normal
        else:
            self.preprocess(size=197988)
            self.make_train_val_partitions()
            self._actual_data_generator = self._actual_data_generator_preprocess

        # Synthetic
        # self.train_val_nsamples = 10000
        # self.train_nsamples = None
        # self.make_train_val_partitions()
        # self.src_embeddings = random.random((self.train_val_nsamples, 1, self.max_sentence, self.embedl)).astype(dtype=self.dtype)
        # self.tgt_embeddings = random.random((self.train_val_nsamples, 1, self.max_sentence, self.embedl)).astype(dtype=self.dtype)

    def load_data(self):
        self.dictionary = self.get_dictionary(self.lang)
        self.mask = self.dictionary("Mask")[0]
        self.mask = np.zeros(self.mask.vector.shape, dtype=self.dtype)
        file = open(self.train_path, "r")
        self.lines = file.readlines()
        file.close()
        self.lines = [line.replace("\n", "") for line in self.lines]

        self.train_val_nsamples = len(self.lines)
        print(self.train_val_nsamples)
        self.train_nsamples = None

    def get_dictionary(self, language):
        import spacy
        table = {
            "en": "en_core_web_md",
            "de": "de_core_news_md"
        }
        if language in table:
            language = table[language]
        return spacy.load(language)

    def make_train_val_partitions(self):
        val_split = self.model.validation_split
        if self.train_nsamples == None:
            s = np.arange(self.train_val_nsamples)
            if self.model.augment_shuffle:
                random.shuffle(s)
            self.train_nsamples = int(self.train_val_nsamples * (1-val_split) // 1)
            self.train_indices = s[:self.train_nsamples]
            self.val_indices = s[self.train_nsamples:]
            self.val_nsamples = len(self.val_indices)
            self.test_nsamples = self.val_nsamples

    def _actual_data_generator_normal(self, part):
        batch_size = self.model.batch_size
        rank = self.model.rank

        for i in range(self.train_val_nsamples // batch_size):
            window = (i * batch_size + rank * batch_size, i * batch_size + (rank + 1) * batch_size)
            src_embeddings = np.zeros((batch_size, 1, self.max_sentence, self.embedl), dtype=self.dtype)
            tgt_embeddings = np.zeros((batch_size, 1, self.max_sentence, self.embedl), dtype=self.dtype)
            for i, doc in enumerate(self.dictionary.pipe(self.lines[window[0]:window[1]])):
                mask = np.random.randint(0, len(doc))
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

    def _actual_data_generator_preprocess(self):
        batch_size = self.model.batch_size
        rank = self.model.rank

        for i in range(self.train_val_nsamples // batch_size):
            window = (i * batch_size + rank * batch_size, i * batch_size + (rank + 1) * batch_size)
            x = self.src_embeddings[window[0]:window[1]]
            y = self.tgt_embeddings[window[0]:window[1]]
            yield x, y

    def _synthetic_data_generator(self):
        batch_size = self.model.batch_size
        rank = self.model.rank

        for i in range(self.train_val_nsamples // batch_size):
            window = (i * batch_size + rank * batch_size, i * batch_size + (rank + 1) * batch_size)
            x = self.src_embeddings[window[0]:window[1]]
            y = self.tgt_embeddings[window[0]:window[1]]
            yield x, y

    # === Preprocess ===
    def preprocess(self, size=None):
        if size is None:
            size = len(self.lines)
        self.train_val_nsamples = size
        self.src_embeddings = np.zeros((size, 1, self.max_sentence, self.embedl), dtype=self.dtype)
        self.tgt_embeddings = np.zeros((size, 1, self.max_sentence, self.embedl), dtype=self.dtype)
        for i, doc in enumerate(self.dictionary.pipe(self.lines[0:size])):
            mask = np.random.randint(0, len(doc))
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
