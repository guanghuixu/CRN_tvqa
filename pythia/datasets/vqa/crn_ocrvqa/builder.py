# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.vqa.crn_ocrvqa.dataset import CRNOCRVQADataset
from pythia.datasets.vqa.crn_textvqa.builder import CRNTextVQABuilder


@Registry.register_builder("crn_ocrvqa")
class CRNOCRVQABuilder(CRNTextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "crn_ocrvqa"
        self.set_dataset_class(CRNOCRVQADataset)
