# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.vqa.crn_stvqa.dataset import CRNSTVQADataset
from pythia.datasets.vqa.crn_textvqa.builder import CRNTextVQABuilder


@Registry.register_builder("crn_stvqa")
class CRNSTVQABuilder(CRNTextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "crn_stvqa"
        self.set_dataset_class(CRNSTVQADataset)
