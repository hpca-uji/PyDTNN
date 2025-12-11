from pathlib import Path
import tarfile
from typing import TYPE_CHECKING

import numpy as np
import csv

from itertools import chain
from pydtnn.datasets.dataset import Dataset
from pydtnn.utils import random

if TYPE_CHECKING:
    from pydtnn.model import Model


TRAIN_NSAMPLES = 86524
TEST_NSAMPLES = 25596
INPUT_SHAPE = (1, 1024, 1024)
OUTPUT_SHAPE = (15,)

CSV_DELIMETER = ','
CSV_LABELS_DELIMETER = '|'
CSV_IMAGES_FIELD = "Image Index"
CSV_LABELS_FIELD = "Finding Labels"
type Class = np.ndarray


def get_dict_file_labels(path: Path) -> dict[str, list[str]]:
    with open(file=path, mode="r") as file:
        reader = csv.DictReader(file, delimiter=CSV_DELIMETER)
        dict_file_labels = dict[str, list[str]]()
        for row in reader:
            image = row[CSV_IMAGES_FIELD]
            labels = row[CSV_LABELS_FIELD].split(CSV_LABELS_DELIMETER)
            dict_file_labels[image] = labels
    return dict_file_labels


class ChestXRay(Dataset):
    """
    ChestXRay Dataset

    The NIH Chest X-ray dataset consists of 100,000 de-identified
    images of chest x-rays. The images are in PNG format.

    Source (SHA1): https://nihcc.app.box.com/v/ChestXray-NIHCC
    48a9f849a8f100a0f1721b33bdbd209767656111 Data_Entry_2017_v2020.csv
    41b85e218abec560a2f5999acbcf333b0f2fa495 test_list.txt
    e3e1b677c01d28481777f3d84e10fcdaac05694c train_val_list.txt
    fef95a7a789bcb0013fbf966cb92c4d92c90becd images_01.tar.gz https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz
    23d2f2cd62d0271b16869abcdd8a00a1fd2492b5 images_02.tar.gz https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz
    69935ac7886c18246446899cb2b75443195847c4 images_03.tar.gz https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz
    fb86b3adaad0e9ff1405154cf6521e180063af10 images_04.tar.gz https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz
    baa8155f0285edb4a07e717f79682713416eb205 images_05.tar.gz https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz
    3a4252d82143757600885121bb57b0ef4e482532 images_06.tar.gz https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz
    cd3cd855acb4e12ca11608be6aae99414d4bc22b images_07.tar.gz https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz
    d8891e0079e88fc04dab45253b86a2214ff499b6 images_08.tar.gz https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz
    84661300d777e07be9ae7d5f37fb82721202f0bc images_09.tar.gz https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz
    30216f59778f259db91d77bcd3d0495c8fce88ef images_10.tar.gz https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz
    97985118ba36f18c27d62371d28c1698478cecfa images_11.tar.gz https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz
    cb2865369f434a9deea11e2d5222b8472890681b images_12.tar.gz https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz

    Normalize (z-score):
    offset: -0.509
    scale:  +4.002
    """

    def __init__(self, model: "Model", force_test_as_validation=False, debug=False):
        super().__init__(model, TRAIN_NSAMPLES, TEST_NSAMPLES, INPUT_SHAPE, OUTPUT_SHAPE, force_test_as_validation=force_test_as_validation, debug=debug)

    def _get_labels(self, image_file: str) -> Class:
        labels = self._dict_images_labels[image_file]
        mask = np.zeros(self.output_shape, dtype=np.uint8)
        for label in labels:
            mask[self.labels2classes[label]] = 1
        return mask

    def _init_actual_data(self):
        self._xy_filenames: list[list[tuple[str, Class]]] = [[("", np.empty((0,)))] for _ in Dataset.Part]
        self.files = Path(self.model.dataset_path)
        csv = self.files.joinpath("Data_Entry_2017_v2020.csv")
        train = self.files.joinpath("train_val_list.txt").read_text().splitlines()
        test = self.files.joinpath("test_list.txt").read_text().splitlines()
        self._dict_images_labels = get_dict_file_labels(csv)

        output_shape = len(set(chain.from_iterable(self._dict_images_labels.values())))

        if output_shape != OUTPUT_SHAPE[0]:
            raise ValueError(f"Mismatch class shape (got: {output_shape}, expect: {OUTPUT_SHAPE[0]})")

        if len(train) != TRAIN_NSAMPLES:
            raise ValueError(f"Mismatch train samples (got: {len(train)}, expect: {TRAIN_NSAMPLES})")

        if len(test) != TEST_NSAMPLES:
            raise ValueError(f"Mismatch train samples (got: {len(test)}, expect: {TEST_NSAMPLES})")

        # Getting the labels and equivalence class - label
        labels = sorted(list({elem for list_elems in self._dict_images_labels.values() for elem in list_elems}))
        self.labels2classes = {labels[_class]: _class for _class in range(len(labels))}

        # Get image map
        self._src_filename = dict[str, str]()
        for gz in self.files.glob("*.gz"):
            with self._gzip_open(str(gz)) as g:
                with tarfile.TarFile(fileobj=g) as t:
                    for member in t.getmembers():
                        if not member.isfile():
                            continue
                        self._src_filename[member.name] = str(gz)

        # Format: 'datasets/chestxray/images_06.tar.gz', 'images/00011575_000.png'
        self._xy_filenames[Dataset.Part.TRAIN] = [(f"images/{name}", self._get_labels(name)) for name in train]
        self._xy_filenames[Dataset.Part.TEST] = [(f"images/{name}", self._get_labels(name)) for name in test]
        self._xy_filenames[Dataset.Part.VAL] = self._xy_filenames[Dataset.Part.TEST] if self.model.test_as_validation else self._xy_filenames[Dataset.Part.TRAIN]

    def _actual_data_generator(self, part):
        offset = self._local_offset[part]
        nsamples = self._local_nsamples[part]
        xy_filenames = self._xy_filenames[part]

        if part is Dataset.Part.TRAIN and self.model.augment_shuffle:
            random.shuffle(xy_filenames)  # type: ignore (numpy shuffle's typing wasn't well defined.)

        xy_filenames = xy_filenames[offset:offset + nsamples]

        for path, y in xy_filenames:
            src_path = self._src_filename[path]
            with self._gzip_open(src_path) as g, tarfile.TarFile(fileobj=g) as t, t.extractfile(path) as fp:
                x = self._load_gray_image(fp)

            # Add N dimension
            x = x[None, ...]
            y = y[None, ...]

            # Set tensor format
            x = self.model.encode_tensor(x)

            # Set dtype and order
            x = x.astype(dtype=self.model.dtype, order="C")
            y = y.astype(dtype=self.model.dtype, order="C")

            # Inplace normalization
            x /= 255.0

            yield x, y
