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


class IWSLT(Dataset):
    """
    IWSLT Dataset

    NOTE: Source unclear
    TODO: Load original dataset

    The IWSLT 2017 Multilingual Task addresses text translation,
    including zero-shot translation, with a single MT system across
    all directions including English, German, Dutch, Italian and
    Romanian. As unofficial task, conventional bilingual text
    translation is offered between English and Arabic, French,
    Japanese, Chinese, German and Korean.

    Source (SHA1): https://huggingface.co/datasets/IWSLT/iwslt2017
    ac7581060e37d197d7fd2f09bb3bf1f8a2d57ddb iwslt_corto.txt
    a3166ce94db1e8e1cff3757f927c7f821229cfe6 iwslt.txt
    8f3d50b3ca9bae54e5cda80d84e5cdb5e33a4b4a wiki_split.txt

    Normalize (z-score):
    offset: ???
    scale:  ???
    """

    def __init__(self, model: "Model", embedl=512, max_sentence=512, split_token="<translation>", force_test_as_validation=False, debug=False):
        super().__init__(model, TRAIN_NSAMPLES, TEST_NSAMPLES, INPUT_SHAPE, OUTPUT_SHAPE, force_test_as_validation=force_test_as_validation, debug=debug)

        self.split_token = split_token
        self.max_sentence = max_sentence
        self.embedl = embedl
        self.train_path = self.model.dataset_path
        self.test_path = self.model.dataset_path
        self.test_as_validation = force_test_as_validation
        self.dtype = self.model.dtype
        self.model = model
        self.lang1 = self.model.dataset_lang
        self.lang2 = self.model.dataset_lang2

    def _init_actual_data(self):
        # Actual
        self.load_data(self.train_path, self.lang1, self.lang2)
        self.make_train_val_partitions()

        # Synthetic
        # self.train_val_nsamples = 10000
        # self.train_nsamples = None
        # self.lines1 = self.lines2 = self.lines1_train = self.lines2_train = self.lines1_val = self.lines2_val = self.lines1_test = self.lines2_test = None
        # self.make_train_val_partitions()
        # # self.src_embeddings = random.random(*self.train_val_nsamples, 1, self.max_sentence, self.embedl)).astype(dtype=self.dtype)
        # # self.tgt_embeddings = random.random(*self.train_val_nsamples, 1, self.max_sentence, self.embedl)).astype(dtype=self.dtype)
        # # self.src_mask = np.zeros((self.train_val_nsamples,1,self.max_sentence),dtype=bool)
        # # self.tgt_mask = np.zeros((self.train_val_nsamples,1,self.max_sentence),dtype=bool)
        # self.src_embeddings = random.random((1000, 1, self.max_sentence, self.embedl)).astype(dtype=self.dtype)
        # self.tgt_embeddings = random.random((1000, 1, self.max_sentence, self.embedl)).astype(dtype=self.dtype)
        # # self.src_mask = np.zeros((1000,1,self.max_sentence),dtype=bool)
        # # self.tgt_mask = np.zeros((1000,1,self.max_sentence),dtype=bool)

    def load_data(self, file, lang1, lang2):
        self.dictionary1 = self.get_dictionary(lang1)
        self.dictionary2 = self.get_dictionary(lang2)
        file = open(self.train_path, "r")
        lines = file.readlines()
        file.close()
        lines = [line.replace("\n", "") for line in lines]
        self.train_val_nsamples = len(lines)
        self.train_nsamples = None
        self.lines1 = [line.split(self.split_token)[0] for line in lines]
        self.lines2 = [line.split(self.split_token)[1] for line in lines]
        # sos = None # <sos>
        # eos = None # <eos>
        # pad = ' '
        # if sos is not None and eos is not None: # No encuentro los eos y sos en spacy
        #     sos = dictionary1(sos).vector
        #     eos = dictionary1(eos).vector
        # pad = dictionary(pad).vector  # No hace falta si inicializamos a 0s

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
            # Make test partition
            if self.test_as_validation:
                self.test_indices = self.val_indices
            # print(len(self.train_indices), len(self.val_indices))
            if self.lines1 is not None:
                self.lines1_train = [self.lines1[i] for i in self.train_indices]
                self.lines2_train = [self.lines2[i] for i in self.train_indices]
                self.lines1_val = [self.lines1[i] for i in self.val_indices]
                self.lines2_val = [self.lines2[i] for i in self.val_indices]
                if self.test_as_validation:
                    self.lines1_test = self.lines1_val
                    self.lines2_test = self.lines2_val
                else:
                    self.lines1_test = [self.lines1[i] for i in self.test_indices]
                    self.lines2_test = [self.lines2[i] for i in self.test_indices]

    def _actual_data_generator(self, part):
        match part:
            case Dataset.Part.TRAIN:
                lines1, lines2 = self.lines1_train, self.lines2_train
            case Dataset.Part.VAL:
                lines1, lines2 = self.lines1_val, self.lines2_val
            case Dataset.Part.TEST:
                lines1, lines2 = self.lines1_test, self.lines2_test

        batch_size = self.model.batch_size

        for i in range(self.train_val_nsamples // batch_size):
            window = (i*batch_size, (i+1)*batch_size)
            src_embeddings = np.zeros((batch_size, self.max_sentence, self.embedl), dtype=np.float32)
            tgt_embeddings = np.zeros((batch_size, self.max_sentence, self.embedl), dtype=np.float32)
            src_mask = np.zeros((batch_size, 1, self.max_sentence), dtype=bool)
            tgt_mask = np.zeros((batch_size, self.max_sentence, self.max_sentence), dtype=bool)
            for i, doc in enumerate(self.dictionary1.pipe(lines1[window[0]:window[1]])):
                for j, word in enumerate(doc):
                    src_embeddings[i, j] = word.vector
                    src_mask[i, 0, j] = 1
            for i, doc in enumerate(self.dictionary2.pipe(lines2[window[0]:window[1]])):
                for j, word in enumerate(doc):
                    tgt_embeddings[i, j] = word.vector
                    tgt_mask[i, j, 0:j+1] = [1] * (j+1)

            x = [src_embeddings, src_mask, tgt_embeddings, tgt_mask]
            y = tgt_embeddings
            yield x, y

    def _synthetic_data_generator(self):
        batch_size = self.model.batch_size
        rank = self.model.rank

        for i in range(self.train_nsamples//batch_size):
            # window = (i * batch_size + rank * batch_size, i * batch_size + (rank + 1) * batch_size)
            window = (0 * batch_size + rank * batch_size, 0 * batch_size + (rank + 1) * batch_size)
            # x = [self.src_embeddings[window[0]:window[1]], self.src_mask[window[0]:window[1]], self.tgt_embeddings[window[0]:window[1]], self.tgt_mask[window[0]:window[1]]]
            # y = self.tgt_embeddings[window[0]:window[1]]
            x = [self.src_embeddings[window[0]:window[1]], self.tgt_embeddings[window[0]:window[1]]]
            y = self.tgt_embeddings[window[0]:window[1]]
            yield x, y

    # === Preprocess ===
    def preprocess(self, original_file_lang1, lang1, original_file_lang2, lang2, destination_file="IWSLT.txt"):
        self.load_data(destination_file, lang1, lang2)
        file = open(original_file_lang1, "r")
        lines1 = file.readlines()
        file.close()
        file = open(original_file_lang2, "r")
        lines2 = file.readlines()
        file.close()
        if len(lines1) != len(lines2):
            print("Los archivos tienen un numero de muestras distintas, {} y {}".format(len(lines1), len(lines2)))
            return -1

        file = open(self.file, "w")
        for i in range(len(lines1)):
            line1 = lines1[i].replace("\n", "")
            line2 = lines2[i].replace("\n", "")
            if len(self.dictionary1(line1)) <= self.max_sentence and len(self.dictionary2(line2)) <= self.max_sentence:
                file.write(line1 + self.split_token + line2 + "\n")
        file.close()

    def xml_to_txt(self, file_in, file_out):
        file = open(file_in, "r")
        # lines = file.read().splitlines()
        lines = file.readlines()
        file.close()
        nlines_in = len(lines)
        lines = [line.replace("\n", "") for line in lines if "<" not in line]
        file = open(file_out, "w")
        file.writelines("\n".join(lines))
        file.close()
        nlines_out = len(lines)
        print("{} lines in, {} lines out".format(nlines_in, nlines_out))
    # === Preprocess ===
