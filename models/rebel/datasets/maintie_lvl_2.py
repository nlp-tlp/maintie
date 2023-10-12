# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""MaintIE: RE dataset."""

from __future__ import absolute_import, division, print_function

import json
import logging

import datasets


_DESCRIPTION = """MaintIE is made of sentences from maintenance texts, annotated with 200+ entity types and six relation types."""

_URL = ""
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "val.json",
    "test": _URL + "test.json",
}

mapping = {
    "isA": "is a",
    "contains": "contains",
    "hasPart": "has part",
    "hasParticipant": "has participant",
    "hasPatient": "has patient",
    "hasAgent": "has agent",
    "hasProperty": "has property",
}


mapping_types = {
    "PhysicalObject/CoveringObject": "<covering object>",
    "PhysicalObject/Substance": "<substance>",
    "PhysicalObject/GuidingObject": "<guiding object>",
    "State/DesirableState": "<desirable state>",
    "PhysicalObject/GeneratingObject": "<generating object>",
    "PhysicalObject/TransformingObject": "<transforming object>",
    "PhysicalObject/MatterProcessingObject": "<matter processing object>",
    "Process": "<process>",
    "Property/UndesirableProperty": "<undesirable property>",
    "State": "<state>",
    "Process/UndesirableProcess": "<undesirable process>",
    "PhysicalObject/InterfacingObject": "<interfacing object>",
    "PhysicalObject/StoringObject": "<storing object>",
    "PhysicalObject/EmittingObject": "<emitting object>",
    "PhysicalObject/PresentingObject": "<presenting object>",
    "Activity/MaintenanceActivity": "<maintenance activity>",
    "PhysicalObject/RestrictingObject": "<restricting object>",
    "Activity/SupportingActivity": "<supporting activity>",
    "Property/DesirableProperty": "<desirable property>",
    "PhysicalObject/ControllingObject": "<controlling object>",
    "Property": "<property>",
    "PhysicalObject/HumanInteractionObject": "<human interaction object>",
    "Activity": "<activity>",
    "PhysicalObject/DrivingObject": "<driving object>",
    "State/UndesirableState": "<undesirable state>",
    "PhysicalObject/InformationProcessingObject": "<information processing object>",
    "PhysicalObject": "<physical object>",
    "PhysicalObject/Organism": "<organism>",
    "PhysicalObject/HoldingObject": "<holding object>",
    "Process/DesirableProcess": "<desirable process>",
    "PhysicalObject/SensingObject": "<sensing object>",
    "PhysicalObject/ProtectingObject": "<protecting object>",
}


class MAINTIEConfig(datasets.BuilderConfig):
    """BuilderConfig for MAINTIE."""

    def __init__(self, **kwargs):
        """BuilderConfig for MAINTIE.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MAINTIEConfig, self).__init__(**kwargs)


class MAINTIE(datasets.GeneratorBasedBuilder):
    """MAINTIE"""

    BUILDER_CONFIGS = [
        MAINTIEConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "triplets": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="",
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files:
            # print("self.config.data_files", self.config.data_files)
            downloaded_files = {
                "train": self.config.data_files[
                    "train"
                ],  # self.config.data_dir + "en_train.jsonl",
                "dev": self.config.data_files[
                    "dev"
                ],  # self.config.data_dir + "en_val.jsonl",
                "test": self.config.data_files[
                    "test"
                ],  # self.config.data_dir + "en_test.jsonl",
            }
        else:
            downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) triplet form."""
        logging.info(
            "generating examples from = %s", filepath[0]
        )  # was filepath which was array of strings.

        with open(filepath[0]) as json_file:
            f = json.load(json_file)

            for id_, row in enumerate(f):
                # print("row:", row)

                triplets = ""
                prev_head = None
                for relation in row["relations"]:
                    if prev_head == relation["head"]:
                        triplets += (
                            f' {mapping_types[row["entities"][relation["head"]]["type"]]} '
                            + " ".join(
                                row["tokens"][
                                    row["entities"][relation["tail"]]["start"] : row[
                                        "entities"
                                    ][relation["tail"]]["end"]
                                ]
                            )
                            + f' {mapping_types[row["entities"][relation["tail"]]["type"]]} '
                            + mapping[relation["type"]]
                        )
                    elif prev_head == None:
                        triplets += (
                            "<triplet> "
                            + " ".join(
                                row["tokens"][
                                    row["entities"][relation["head"]]["start"] : row[
                                        "entities"
                                    ][relation["head"]]["end"]
                                ]
                            )
                            + f' {mapping_types[row["entities"][relation["head"]]["type"]]} '
                            + " ".join(
                                row["tokens"][
                                    row["entities"][relation["tail"]]["start"] : row[
                                        "entities"
                                    ][relation["tail"]]["end"]
                                ]
                            )
                            + f' {mapping_types[row["entities"][relation["tail"]]["type"]]} '
                            + mapping[relation["type"]]
                        )
                        prev_head = relation["head"]
                    else:
                        triplets += (
                            " <triplet> "
                            + " ".join(
                                row["tokens"][
                                    row["entities"][relation["head"]]["start"] : row[
                                        "entities"
                                    ][relation["head"]]["end"]
                                ]
                            )
                            + f' {mapping_types[row["entities"][relation["head"]]["type"]]} '
                            + " ".join(
                                row["tokens"][
                                    row["entities"][relation["tail"]]["start"] : row[
                                        "entities"
                                    ][relation["tail"]]["end"]
                                ]
                            )
                            + f' {mapping_types[row["entities"][relation["tail"]]["type"]]} '
                            + mapping[relation["type"]]
                        )
                        prev_head = relation["head"]
                text = " ".join(row["tokens"])

                yield str(id_), {
                    "title": str(id_),
                    "context": text,
                    "id": str(id_),
                    "triplets": triplets,
                }
