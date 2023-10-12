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
    "PhysicalObject": "<physical object>",
    "PhysicalObject/Substance": "<substance>",
    "PhysicalObject/Substance/Gas": "<gas>",
    "PhysicalObject/Substance/Liquid": "<liquid>",
    "PhysicalObject/Substance/Solid": "<solid>",
    "PhysicalObject/Substance/Mixture": "<mixture>",
    "PhysicalObject/Organism": "<organism>",
    "PhysicalObject/Organism/Person": "<person>",
    "PhysicalObject/SensingObject": "<sensing object>",
    "PhysicalObject/SensingObject/ElectricPotentialSensingObject": "<electric potential sensing object>",
    "PhysicalObject/SensingObject/ResistivitySensingObject": "<resistivity sensing object>",
    "PhysicalObject/SensingObject/ElectricCurrentSensingObject": "<electric current sensing object>",
    "PhysicalObject/SensingObject/DensitySensingObject": "<density sensing object>",
    "PhysicalObject/SensingObject/FieldSensingObject": "<field sensing object>",
    "PhysicalObject/SensingObject/FlowSensingObject": "<flow sensing object>",
    "PhysicalObject/SensingObject/PhysicalDimensionSensingObject": "<physical dimension sensing object>",
    "PhysicalObject/SensingObject/EnergySensingObject": "<energy sensing object>",
    "PhysicalObject/SensingObject/PowerSensingObject": "<power sensing object>",
    "PhysicalObject/SensingObject/TimeSensingObject": "<time sensing object>",
    "PhysicalObject/SensingObject/LevelSensingObject": "<level sensing object>",
    "PhysicalObject/SensingObject/HumiditySensingObject": "<humidity sensing object>",
    "PhysicalObject/SensingObject/PressureSensingObject": "<pressure sensing object>",
    "PhysicalObject/SensingObject/ConcentrationSensingObject": "<concentration sensing object>",
    "PhysicalObject/SensingObject/RadiationSensingObject": "<radiation sensing object>",
    "PhysicalObject/SensingObject/TimeRatingObject": "<time rating object>",
    "PhysicalObject/SensingObject/TemperatureSensingObject": "<temperature sensing object>",
    "PhysicalObject/SensingObject/MultiQuantitySensingObject": "<multi quantity sensing object>",
    "PhysicalObject/SensingObject/ForceSensingObject": "<force sensing object>",
    "PhysicalObject/SensingObject/AudioVisualSensingObject": "<audio visual sensing object>",
    "PhysicalObject/SensingObject/InformationSensingObject": "<information sensing object>",
    "PhysicalObject/SensingObject/IncidentSensingObject": "<incident sensing object>",
    "PhysicalObject/StoringObject": "<storing object>",
    "PhysicalObject/StoringObject/CapacitiveStoringObject": "<capacitive storing object>",
    "PhysicalObject/StoringObject/InductiveStoringObject": "<inductive storing object>",
    "PhysicalObject/StoringObject/ElectrochemicalStoringObject": "<electrochemical storing object>",
    "PhysicalObject/StoringObject/InformationStoringObject": "<information storing object>",
    "PhysicalObject/StoringObject/OpenStationaryStoringObject": "<open stationary storing object>",
    "PhysicalObject/StoringObject/EnclosedStationaryStoringObject": "<enclosed stationary storing object>",
    "PhysicalObject/StoringObject/MoveableStoringObject": "<moveable storing object>",
    "PhysicalObject/StoringObject/ThermalEnergyStoringObject": "<thermal energy storing object>",
    "PhysicalObject/StoringObject/MechanicalEnergyStoringObject": "<mechanical energy storing object>",
    "PhysicalObject/EmittingObject": "<emitting object>",
    "PhysicalObject/EmittingObject/LightObject": "<light object>",
    "PhysicalObject/EmittingObject/ElectricHeatingObject": "<electric heating object>",
    "PhysicalObject/EmittingObject/ElectricCoolingObject": "<electric cooling object>",
    "PhysicalObject/EmittingObject/WirelessPowerObject": "<wireless power object>",
    "PhysicalObject/EmittingObject/ThermalEnergyTransferObject": "<thermal energy transfer object>",
    "PhysicalObject/EmittingObject/CombustionHeatingObject": "<combustion heating object>",
    "PhysicalObject/EmittingObject/ThermalHeatingObject": "<thermal heating object>",
    "PhysicalObject/EmittingObject/ThermalCoolingObject": "<thermal cooling object>",
    "PhysicalObject/EmittingObject/NuclearPoweredHeatingObject": "<nuclear powered heating object>",
    "PhysicalObject/EmittingObject/ParticleEmittingObject": "<particle emitting object>",
    "PhysicalObject/EmittingObject/AcousticWaveEmittingObject": "<acoustic wave emitting object>",
    "PhysicalObject/ProtectingObject": "<protecting object>",
    "PhysicalObject/ProtectingObject/OvervoltageProtectingObject": "<overvoltage protecting object>",
    "PhysicalObject/ProtectingObject/EarthFaultCurrentProtectingObject": "<earth fault current protecting object>",
    "PhysicalObject/ProtectingObject/OvercurrentProtectingObject": "<overcurrent protecting object>",
    "PhysicalObject/ProtectingObject/FieldProtectingObject": "<field protecting object>",
    "PhysicalObject/ProtectingObject/PressureProtectingObject": "<pressure protecting object>",
    "PhysicalObject/ProtectingObject/FireProtectingObject": "<fire protecting object>",
    "PhysicalObject/ProtectingObject/MechanicalForceProtectingObject": "<mechanical force protecting object>",
    "PhysicalObject/ProtectingObject/PreventiveProtectingObject": "<preventive protecting object>",
    "PhysicalObject/ProtectingObject/WearProtectingObject": "<wear protecting object>",
    "PhysicalObject/ProtectingObject/EnvironmentProtectingObject": "<environment protecting object>",
    "PhysicalObject/ProtectingObject/TemperatureProtectingObject": "<temperature protecting object>",
    "PhysicalObject/GeneratingObject": "<generating object>",
    "PhysicalObject/GeneratingObject/MechanicalToElectricalEnergyGeneratingObject": "<mechanical to electrical energy generating object>",
    "PhysicalObject/GeneratingObject/ChemicalToElectricalEnergyGeneratingObject": "<chemical to electrical energy generating object>",
    "PhysicalObject/GeneratingObject/SolarToElectricalEnergyGeneratingObject": "<solar to electrical energy generating object>",
    "PhysicalObject/GeneratingObject/SignalGeneratingObject": "<signal generating object>",
    "PhysicalObject/GeneratingObject/ContinuousTransferObject": "<continuous transfer object>",
    "PhysicalObject/GeneratingObject/DiscontinuousTransferObject": "<discontinuous transfer object>",
    "PhysicalObject/GeneratingObject/LiquidFlowGeneratingObject": "<liquid flow generating object>",
    "PhysicalObject/GeneratingObject/GaseousFlowGeneratingObject": "<gaseous flow generating object>",
    "PhysicalObject/GeneratingObject/SolarToThermalEnergyGeneratingObject": "<solar to thermal energy generating object>",
    "PhysicalObject/MatterProcessingObject": "<matter processing object>",
    "PhysicalObject/MatterProcessingObject/PrimaryFormingObject": "<primary forming object>",
    "PhysicalObject/MatterProcessingObject/SurfaceTreatmentObject": "<surface treatment object>",
    "PhysicalObject/MatterProcessingObject/AssemblingObject": "<assembling object>",
    "PhysicalObject/MatterProcessingObject/ForceSeparatingObject": "<force separating object>",
    "PhysicalObject/MatterProcessingObject/ThermalSeparatingObject": "<thermal separating object>",
    "PhysicalObject/MatterProcessingObject/MechanicalSeparatingObject": "<mechanical separating object>",
    "PhysicalObject/MatterProcessingObject/ElectricOrMagneticSeparatingObject": "<electric or magnetic separating object>",
    "PhysicalObject/MatterProcessingObject/ChemicalSeparatingObject": "<chemical separating object>",
    "PhysicalObject/MatterProcessingObject/GrindingAndCrushingObject": "<grinding and crushing object>",
    "PhysicalObject/MatterProcessingObject/AgglomeratingObject": "<agglomerating object>",
    "PhysicalObject/MatterProcessingObject/MixingObject": "<mixing object>",
    "PhysicalObject/MatterProcessingObject/ReactingObject": "<reacting object>",
    "PhysicalObject/InformationProcessingObject": "<information processing object>",
    "PhysicalObject/InformationProcessingObject/ElectricSignalProcessingObject": "<electric signal processing object>",
    "PhysicalObject/InformationProcessingObject/ElectricSignalRelayingObject": "<electric signal relaying object>",
    "PhysicalObject/InformationProcessingObject/OpticalSignallingObject": "<optical signalling object>",
    "PhysicalObject/InformationProcessingObject/FluidSignallingObject": "<fluid signalling object>",
    "PhysicalObject/InformationProcessingObject/MechanicalSignallingObject": "<mechanical signalling object>",
    "PhysicalObject/InformationProcessingObject/MultipleKindSignallingObject": "<multiple kind signalling object>",
    "PhysicalObject/DrivingObject": "<driving object>",
    "PhysicalObject/DrivingObject/ElectromagneticRotationalDrivingObject": "<electromagnetic rotational driving object>",
    "PhysicalObject/DrivingObject/ElectromagneticLinearDrivingObject": "<electromagnetic linear driving object>",
    "PhysicalObject/DrivingObject/MagneticForceDrivingObject": "<magnetic force driving object>",
    "PhysicalObject/DrivingObject/PiezoelectricDrivingObject": "<piezoelectric driving object>",
    "PhysicalObject/DrivingObject/MechanicalEnergyDrivingObject": "<mechanical energy driving object>",
    "PhysicalObject/DrivingObject/FluidPoweredDrivingObject": "<fluid powered driving object>",
    "PhysicalObject/DrivingObject/CombustionEngine": "<combustion engine>",
    "PhysicalObject/DrivingObject/HeatEngine": "<heat engine>",
    "PhysicalObject/CoveringObject": "<covering object>",
    "PhysicalObject/CoveringObject/InfillingObject": "<infilling object>",
    "PhysicalObject/CoveringObject/ClosureObject": "<closure object>",
    "PhysicalObject/CoveringObject/FinishingObject": "<finishing object>",
    "PhysicalObject/CoveringObject/TerminatingObject": "<terminating object>",
    "PhysicalObject/CoveringObject/HidingObject": "<hiding object>",
    "PhysicalObject/PresentingObject": "<presenting object>",
    "PhysicalObject/PresentingObject/VisibleStateIndicator": "<visible state indicator>",
    "PhysicalObject/PresentingObject/ScalarDisplay": "<scalar display>",
    "PhysicalObject/PresentingObject/GraphicalDisplay": "<graphical display>",
    "PhysicalObject/PresentingObject/AcousticDevice": "<acoustic device>",
    "PhysicalObject/PresentingObject/TactileDevice": "<tactile device>",
    "PhysicalObject/PresentingObject/OrnamentalObject": "<ornamental object>",
    "PhysicalObject/PresentingObject/MultipleFormPresentingObject": "<multiple form presenting object>",
    "PhysicalObject/ControllingObject": "<controlling object>",
    "PhysicalObject/ControllingObject/ElectricControllingObject": "<electric controlling object>",
    "PhysicalObject/ControllingObject/ElectricSeparatingObject": "<electric separating object>",
    "PhysicalObject/ControllingObject/ElectricEarthingObject": "<electric earthing object>",
    "PhysicalObject/ControllingObject/SealedFluidSwitchingObject": "<sealed fluid switching object>",
    "PhysicalObject/ControllingObject/SealedFluidVaryingObject": "<sealed fluid varying object>",
    "PhysicalObject/ControllingObject/OpenFlowControllingObject": "<open flow controlling object>",
    "PhysicalObject/ControllingObject/SpaceAccessObject": "<space access object>",
    "PhysicalObject/ControllingObject/SolidSubstanceFlowVaryingObject": "<solid substance flow varying object>",
    "PhysicalObject/ControllingObject/MechanicalMovementControllingObject": "<mechanical movement controlling object>",
    "PhysicalObject/ControllingObject/MultipleMeasureControllingObject": "<multiple measure controlling object>",
    "PhysicalObject/RestrictingObject": "<restricting object>",
    "PhysicalObject/RestrictingObject/ElectricityRestrictingObject": "<electricity restricting object>",
    "PhysicalObject/RestrictingObject/ElectricityStabilisingObject": "<electricity stabilising object>",
    "PhysicalObject/RestrictingObject/SignalStabilisingObject": "<signal stabilising object>",
    "PhysicalObject/RestrictingObject/MovementRestrictingObject": "<movement restricting object>",
    "PhysicalObject/RestrictingObject/ReturnFlowRestrictingObject": "<return flow restricting object>",
    "PhysicalObject/RestrictingObject/FlowRestrictor": "<flow restrictor>",
    "PhysicalObject/RestrictingObject/LocalClimateStabilisingObject": "<local climate stabilising object>",
    "PhysicalObject/RestrictingObject/AccessRestrictingObject": "<access restricting object>",
    "PhysicalObject/HumanInteractionObject": "<human interaction object>",
    "PhysicalObject/HumanInteractionObject/FaceInteractionObject": "<face interaction object>",
    "PhysicalObject/HumanInteractionObject/HandInteractionObject": "<hand interaction object>",
    "PhysicalObject/HumanInteractionObject/FootInteractionObject": "<foot interaction object>",
    "PhysicalObject/HumanInteractionObject/FingerInteractionObject": "<finger interaction object>",
    "PhysicalObject/HumanInteractionObject/MovementInteractionObject": "<movement interaction object>",
    "PhysicalObject/HumanInteractionObject/MultiInteractionObject": "<multi interaction object>",
    "PhysicalObject/TransformingObject": "<transforming object>",
    "PhysicalObject/TransformingObject/ElectricEnergyTransformingObject": "<electric energy transforming object>",
    "PhysicalObject/TransformingObject/ElectricEnergyConvertingObject": "<electric energy converting object>",
    "PhysicalObject/TransformingObject/UniversalPowerSupply": "<universal power supply>",
    "PhysicalObject/TransformingObject/SignalConvertingObject": "<signal converting object>",
    "PhysicalObject/TransformingObject/MechanicalEnergyTransformingObject": "<mechanical energy transforming object>",
    "PhysicalObject/TransformingObject/MassReductionObject": "<mass reduction object>",
    "PhysicalObject/TransformingObject/MatterReshapingObject": "<matter reshaping object>",
    "PhysicalObject/TransformingObject/OrganicPlant": "<organic plant>",
    "PhysicalObject/HoldingObject": "<holding object>",
    "PhysicalObject/HoldingObject/PositioningObject": "<positioning object>",
    "PhysicalObject/HoldingObject/CarryingObject": "<carrying object>",
    "PhysicalObject/HoldingObject/EnclosingObject": "<enclosing object>",
    "PhysicalObject/HoldingObject/StructuralSupportingObject": "<structural supporting object>",
    "PhysicalObject/HoldingObject/ReinforcingObject": "<reinforcing object>",
    "PhysicalObject/HoldingObject/FramingObject": "<framing object>",
    "PhysicalObject/HoldingObject/JointingObject": "<jointing object>",
    "PhysicalObject/HoldingObject/FasteningObject": "<fastening object>",
    "PhysicalObject/HoldingObject/LevellingObject": "<levelling object>",
    "PhysicalObject/HoldingObject/ExistingGround": "<existing ground>",
    "PhysicalObject/GuidingObject": "<guiding object>",
    "PhysicalObject/GuidingObject/ElectricEnergyGuidingObject": "<electric energy guiding object>",
    "PhysicalObject/GuidingObject/ReferencingPotentialGuidingObject": "<referencing potential guiding object>",
    "PhysicalObject/GuidingObject/ElectricSignalGuidingObject": "<electric signal guiding object>",
    "PhysicalObject/GuidingObject/LightGuidingObject": "<light guiding object>",
    "PhysicalObject/GuidingObject/SoundGuidingObject": "<sound guiding object>",
    "PhysicalObject/GuidingObject/SolidMatterGuidingObject": "<solid matter guiding object>",
    "PhysicalObject/GuidingObject/OpenEnclosureGuidingObject": "<open enclosure guiding object>",
    "PhysicalObject/GuidingObject/ClosedEnclosureGuidingObject": "<closed enclosure guiding object>",
    "PhysicalObject/GuidingObject/MechanicalEnergyGuidingObject": "<mechanical energy guiding object>",
    "PhysicalObject/GuidingObject/RailObject": "<rail object>",
    "PhysicalObject/GuidingObject/ThermalEnergyGuidingObject": "<thermal energy guiding object>",
    "PhysicalObject/GuidingObject/MultipleFlowGuidingObject": "<multiple flow guiding object>",
    "PhysicalObject/GuidingObject/HighVoltageElectricEnergyGuidingObject": "<high voltage electric energy guiding object>",
    "PhysicalObject/GuidingObject/LowVoltageElectricEnergyGuidingObject": "<low voltage electric energy guiding object>",
    "PhysicalObject/InterfacingObject": "<interfacing object>",
    "PhysicalObject/InterfacingObject/HighVoltageConnectingObject": "<high voltage connecting object>",
    "PhysicalObject/InterfacingObject/LowVoltageConnectingObject": "<low voltage connecting object>",
    "PhysicalObject/InterfacingObject/PotentialConnectingObject": "<potential connecting object>",
    "PhysicalObject/InterfacingObject/ElectricSignalConnectingObject": "<electric signal connecting object>",
    "PhysicalObject/InterfacingObject/LightCollectingObject": "<light collecting object>",
    "PhysicalObject/InterfacingObject/CollectingInterfacingObject": "<collecting interfacing object>",
    "PhysicalObject/InterfacingObject/SealedFlowConnectingObject": "<sealed flow connecting object>",
    "PhysicalObject/InterfacingObject/NonDetachableCoupling": "<non detachable coupling>",
    "PhysicalObject/InterfacingObject/DetachableCoupling": "<detachable coupling>",
    "PhysicalObject/InterfacingObject/LevelConnectingObject": "<level connecting object>",
    "PhysicalObject/InterfacingObject/SpaceLinkingObject": "<space linking object>",
    "PhysicalObject/InterfacingObject/MultipleFlowConnectorObject": "<multiple flow connector object>",
    "Activity": "<activity>",
    "Activity/MaintenanceActivity": "<maintenance activity>",
    "Activity/MaintenanceActivity/Adjust": "<adjust>",
    "Activity/MaintenanceActivity/Calibrate": "<calibrate>",
    "Activity/MaintenanceActivity/Diagnose": "<diagnose>",
    "Activity/MaintenanceActivity/Inspect": "<inspect>",
    "Activity/MaintenanceActivity/Replace": "<replace>",
    "Activity/MaintenanceActivity/Repair": "<repair>",
    "Activity/MaintenanceActivity/Service": "<service>",
    "Activity/SupportingActivity": "<supporting activity>",
    "Activity/SupportingActivity/Admin": "<admin>",
    "Activity/SupportingActivity/Assemble": "<assemble>",
    "Activity/SupportingActivity/Isolate": "<isolate>",
    "Activity/SupportingActivity/Measure": "<measure>",
    "Activity/SupportingActivity/Modify": "<modify>",
    "Activity/SupportingActivity/Move": "<move>",
    "Activity/SupportingActivity/Operate": "<operate>",
    "Activity/SupportingActivity/Perform": "<perform>",
    "Activity/SupportingActivity/Teamwork": "<teamwork>",
    "State": "<state>",
    "State/DesirableState": "<desirable state>",
    "State/DesirableState/NormalState": "<normal state>",
    "State/UndesirableState": "<undesirable state>",
    "State/UndesirableState/DegradedState": "<degraded state>",
    "State/UndesirableState/FailedState": "<failed state>",
    "Process": "<process>",
    "Process/DesirableProcess": "<desirable process>",
    "Process/UndesirableProcess": "<undesirable process>",
    "Property": "<property>",
    "Property/DesirableProperty": "<desirable property>",
    "Property/UndesirableProperty": "<undesirable property>",
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
