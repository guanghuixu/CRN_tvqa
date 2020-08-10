# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.vqa.crn_textvqa.dataset import CRNTextVQADataset
from pythia.datasets.vqa.textvqa.builder import TextVQABuilder


@Registry.register_builder("crn_textvqa")
class CRNTextVQABuilder(TextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "crn_textvqa"
        self.set_dataset_class(CRNTextVQADataset)
