# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.datasets.vqa.crn_textvqa.dataset import CRNTextVQADataset


class CRNOCRVQADataset(CRNTextVQADataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(
            dataset_type, imdb_file_index, config, *args, **kwargs
        )
        self._name = "crn_ocrvqa"
