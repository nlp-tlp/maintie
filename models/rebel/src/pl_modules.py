from typing import Any
import nltk
import json
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from score import score, re_score
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from scheduler import get_inverse_square_root_schedule_with_warmup
from datasets import load_dataset, load_metric
from torch.nn.utils.rnn import pad_sequence
from utils import (
    BartTripletHead,
    shift_tokens_left,
    extract_triplets_typed,
    extract_triplets,
    extract_maintie_triplets_typed,
)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
    "inverse_square_root": get_inverse_square_root_schedule_with_warmup,
}


class BasePLModule(pl.LightningModule):
    def __init__(
        self,
        conf,
        config: AutoConfig,
        tokenizer: AutoTokenizer,
        model: AutoModelForSeq2SeqLM,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        if self.model.config.decoder_start_token_id is None:
            raise ValueError(
                "Make sure that `config.decoder_start_token_id` is correctly defined"
            )

        if self.hparams.label_smoothing == 0:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        else:
            # dynamically import label_smoothed_nll_loss
            from utils import label_smoothed_nll_loss

            self.loss_fn = label_smoothed_nll_loss

        # https://github.com/Lightning-AI/lightning/pull/16520
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, inputs, labels, **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        if self.hparams.label_smoothing == 0:
            if self.hparams is not None and self.hparams.ignore_pad_token_for_loss:
                # force training to ignore pad token
                outputs = self.model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=True,
                )
                logits = outputs["logits"]
                loss = self.loss_fn(
                    logits.view(-1, logits.shape[-1]), labels.view(-1)
                )  # , ignore_index=self.config.pad_token_id)
            else:
                # compute usual loss via models
                outputs = self.model(
                    **inputs,
                    labels=labels,
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=True,
                )
                loss = outputs["loss"]
                logits = outputs["logits"]
        else:
            # compute label smoothed loss
            outputs = self.model(
                **inputs, use_cache=False, return_dict=True, output_hidden_states=True
            )
            logits = outputs["logits"]
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            # labels = torch.where(labels != -100, labels, self.config.pad_token_id)
            labels.masked_fill_(labels == -100, self.config.pad_token_id)
            loss, _ = self.loss_fn(
                lprobs,
                labels,
                self.hparams.label_smoothing,
                ignore_index=self.config.pad_token_id,
            )
        output_dict = {"loss": loss, "logits": logits}
        # return loss, logits
        return output_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        labels = batch.pop("labels")
        labels_original = labels.clone()
        batch["decoder_input_ids"] = torch.where(
            labels != -100, labels, self.config.pad_token_id
        )
        labels = shift_tokens_left(labels, -100)
        forward_output = self.forward(batch, labels)
        self.log("loss", forward_output["loss"])
        batch["labels"] = labels_original
        if "loss_aux" in forward_output:
            self.log("loss_classifier", forward_output["loss_aux"])
            return forward_output["loss"] + forward_output["loss_aux"]

        self.train_step_outputs.append(forward_output["loss"])

        return forward_output["loss"]  # + forward_output['loss_aux']

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            self.config.pad_token_id
            if self.config.pad_token_id is not None
            else self.config.eos_token_id
        )

        if pad_token_id is None:
            raise ValueError(
                f"Make sure that either `config.pad_token_id` or `config.eos_token_id` is defined if tensor has to be padded to `max_length`={max_length}"
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def generate_triples(
        self,
        batch,
        labels,
    ) -> None:
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": self.hparams.eval_beams
            if self.hparams.eval_beams is not None
            else self.config.num_beams,
        }

        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            use_cache=True,
            **gen_kwargs,
        )

        decoded_preds = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=False
        )
        decoded_labels = self.tokenizer.batch_decode(
            torch.where(labels != -100, labels, self.config.pad_token_id),
            skip_special_tokens=False,
        )

        if self.hparams.dataset_name.split("/")[-1] == "conll04_typed.py":
            return [extract_triplets_typed(rel) for rel in decoded_preds], [
                extract_triplets_typed(rel) for rel in decoded_labels
            ]

        if "maintie" in self.hparams.dataset_name.split("/")[-1]:
            if "_3" in self.hparams.dataset_name.split("/")[-1]:
                # Level 3 MaintIE data

                maintie_mapping_types = {
                    v: k
                    for k, v in {
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
                    }.items()
                }
            elif "_2" in self.hparams.dataset_name.split("/")[-1]:
                maintie_mapping_types = {
                    v: k
                    for k, v in {
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
                    }.items()
                }
            elif "_1" in self.hparams.dataset_name.split("/")[-1]:
                maintie_mapping_types = {
                    v: k
                    for k, v in {
                        "PhysicalObject": "<physical object>",
                        "Process": "<process>",
                        "Property": "<property>",
                        "Activity": "<activity>",
                        "State": "<state>",
                    }.items()
                }
            elif "_0" in self.hparams.dataset_name.split("/")[-1]:
                # Untyped MaintIE configuration.
                _preds = [extract_triplets(rel) for rel in decoded_preds]
                _gts = [extract_triplets(rel) for rel in decoded_labels]
                return _preds, _gts
            else:
                raise NotImplementedError("MaintIE level not implemented yet!")
            _preds = [
                extract_maintie_triplets_typed(rel, mapping_types=maintie_mapping_types)
                for rel in decoded_preds
            ]

            # print(f'"decoded_preds"\n{decoded_preds[:5]}')

            _gts = [
                extract_maintie_triplets_typed(rel, mapping_types=maintie_mapping_types)
                for rel in decoded_labels
            ]
            # print(f'\n"decoded_labels"\n{decoded_labels[:5]}')

            return _preds, _gts

        return [extract_triplets(rel) for rel in decoded_preds], [
            extract_triplets(rel) for rel in decoded_labels
        ]

    def generate_samples(
        self,
        # model,
        # tokenizer,
        batch,
        labels,
    ) -> None:
        # labels = batch.pop("labels")
        # pick the last batch and logits
        # x, y = batch
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": self.hparams.eval_beams
            if self.hparams.eval_beams is not None
            else self.config.num_beams,
        }
        relation_start = labels == 50265
        relation_start = torch.roll(relation_start, 1, 1)
        relation_start = torch.cumsum(relation_start, dim=1)
        labels_decoder = torch.where(
            relation_start == 1, self.tokenizer.pad_token_id, labels
        )
        labels_decoder[:, -1] = 2
        labels_decoder = torch.roll(labels_decoder, 1, 1)

        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            decoder_input_ids=labels_decoder.to(self.model.device),
            use_cache=False,
            **gen_kwargs,
        )
        relation_start = generated_tokens == 50265
        relation_start = torch.roll(relation_start, 2, 1)

        decoded_preds = self.tokenizer.batch_decode(
            generated_tokens[relation_start == 1], skip_special_tokens=False
        )

        return [rel.strip() for rel in decoded_preds]

    def forward_samples(
        self,
        # model,
        # tokenizer,
        batch,
        labels,
    ) -> None:
        relation_start = labels == 50265
        relation_start = torch.roll(relation_start, 2, 1)
        labels = torch.where(
            torch.cumsum(relation_start, dim=1) == 1,
            self.tokenizer.pad_token_id,
            labels,
        )
        labels[:, -1] = 0
        labels = torch.roll(labels, 1, 1)
        min_padding = min(torch.sum((labels == 1).int(), 1))
        labels_decoder = torch.randint(
            60000, (labels.shape[0], labels.shape[1] - min_padding)
        )
        labels_decoder = labels[:, :-min_padding]
        outputs = self.model(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            decoder_input_ids=labels_decoder.to(self.model.device),
            return_dict=True,
        )
        next_token_logits = outputs.logits[relation_start[:, :-min_padding] == 1]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        decoded_preds = self.tokenizer.batch_decode(
            next_tokens, skip_special_tokens=False
        )

        return [rel.strip() for rel in decoded_preds]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            "length_penalty": 0,
            "num_beams": self.hparams.eval_beams
            if self.hparams.eval_beams is not None
            else self.config.num_beams,
        }

        if self.hparams.predict_with_generate and not self.hparams.prediction_loss_only:
            generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, gen_kwargs["max_length"]
                )

        labels = batch.pop("labels")
        batch["decoder_input_ids"] = torch.where(
            labels != -100, labels, self.config.pad_token_id
        )

        labels = shift_tokens_left(labels, -100)
        with torch.no_grad():
            # compute loss on predict data
            forward_output = self.forward(batch, labels)

        forward_output["loss"] = forward_output["loss"].mean().detach()

        if self.hparams.prediction_loss_only:
            self.log("val_loss", forward_output["loss"])
            return

        forward_output["logits"] = (
            generated_tokens.detach()
            if self.hparams.predict_with_generate
            else forward_output["logits"].detach()
        )

        if labels.shape[-1] < gen_kwargs["max_length"]:
            forward_output["labels"] = self._pad_tensors_to_max_len(
                labels, gen_kwargs["max_length"]
            )
        else:
            forward_output["labels"] = labels

        if self.hparams.predict_with_generate:
            metrics = self.compute_metrics(
                forward_output["logits"].detach().cpu(),
                forward_output["labels"].detach().cpu(),
            )
        else:
            metrics = {}
        metrics["val_loss"] = forward_output["loss"]
        for key in sorted(metrics.keys()):
            self.log(key, metrics[key])

        outputs = {}
        outputs["predictions"], outputs["labels"] = self.generate_triples(batch, labels)

        self.validation_step_outputs.append(outputs)
        return outputs

    def test_step(self, batch: dict, batch_idx: int) -> None:
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            "length_penalty": 0,
            "num_beams": self.hparams.eval_beams
            if self.hparams.eval_beams is not None
            else self.config.num_beams,
        }

        if self.hparams.predict_with_generate and not self.hparams.prediction_loss_only:
            generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, gen_kwargs["max_length"]
                )

        labels = batch.pop("labels")
        batch["decoder_input_ids"] = torch.where(
            labels != -100, labels, self.config.pad_token_id
        )
        labels = shift_tokens_left(labels, -100)
        with torch.no_grad():
            # compute loss on predict data
            forward_output = self.forward(batch, labels)

        forward_output["loss"] = forward_output["loss"].mean().detach()
        if self.hparams.prediction_loss_only:
            self.log("test_loss", forward_output["loss"])
            return

        forward_output["logits"] = (
            generated_tokens.detach()
            if self.hparams.predict_with_generate
            else forward_output["logits"].detach()
        )

        if labels.shape[-1] < gen_kwargs["max_length"]:
            forward_output["labels"] = self._pad_tensors_to_max_len(
                labels, gen_kwargs["max_length"]
            )
        else:
            forward_output["labels"] = labels

        if self.hparams.predict_with_generate:
            metrics = self.compute_metrics(
                forward_output["logits"].detach().cpu(),
                forward_output["labels"].detach().cpu(),
            )
        else:
            metrics = {}
        metrics["test_loss"] = forward_output["loss"]
        for key in sorted(metrics.keys()):
            self.log(key, metrics[key], prog_bar=True)

        if self.hparams.finetune:
            self.test_step_outputs.append(
                {"predictions": self.forward_samples(batch, labels)}
            )
            return {"predictions": self.forward_samples(batch, labels)}

        else:
            outputs = {}
            outputs["predictions"], outputs["labels"] = self.generate_triples(
                batch, labels
            )
            self.test_step_outputs.append(outputs)
            return outputs

    def on_validation_epoch_end(self) -> Any:  # , output: dict) -> Any:
        # print("self.validation_step_outputs\n", self.validation_step_outputs[0])
        # print("self.validation_step_outputs size:", len(self.validation_step_outputs))
        output = self.validation_step_outputs

        if "maintie" in self.hparams.dataset_name.split("/")[-1]:
            # Boundaries evaluation used, not strict. Boundaries == strict when no entity types used.
            scores, precision, recall, f1 = re_score(
                [item for pred in output for item in pred["predictions"]],
                [item for pred in output for item in pred["labels"]],
                [
                    "is a",
                    "contains",
                    "has part",
                    "has participant",
                    "has patient",
                    "has agent",
                    "has property",
                ],
                "boundaries"
                if "_0" in self.hparams.dataset_name.split("/")[-1]
                else "strict",
            )
        else:
            raise NotImplementedError("Dataset not implemented yet")

        self.log("val_prec_micro", precision)
        self.log("val_recall_micro", recall)
        self.log("val_F1_micro", f1)

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> Any:  # , output: dict) -> Any:
        output = self.test_step_outputs

        if "maintie" in self.hparams.dataset_name.split("/")[-1]:
            if "_0" in self.hparams.dataset_name.split("/")[-1]:
                # Only calc boundaries RE - these are the same as strict as its untyped.
                print("CALCULATING UNTYPED RE BOUNDARY SCORES")
                scores, precision, recall, f1 = re_score(
                    [item for pred in output for item in pred["predictions"]],
                    [item for pred in output for item in pred["labels"]],
                    [
                        "is a",
                        "contains",
                        "has part",
                        "has participant",
                        "has patient",
                        "has agent",
                        "has property",
                    ],
                    mode="boundaries",
                )
            else:
                scores, precision, recall, f1 = re_score(
                    [item for pred in output for item in pred["predictions"]],
                    [item for pred in output for item in pred["labels"]],
                    [
                        "is a",
                        "contains",
                        "has part",
                        "has participant",
                        "has patient",
                        "has agent",
                        "has property",
                    ],
                    "strict",
                )

                (
                    boundaries_scores,
                    boundaries_precision,
                    boundaries_recall,
                    boundaries_f1,
                ) = re_score(
                    [item for pred in output for item in pred["predictions"]],
                    [item for pred in output for item in pred["labels"]],
                    [
                        "is a",
                        "contains",
                        "has part",
                        "has participant",
                        "has patient",
                        "has agent",
                        "has property",
                    ],
                    mode="boundaries",
                )
        else:
            raise NotImplementedError("Dataset not implemented yet")

        self.log("test_prec_micro", precision)
        self.log("test_recall_micro", recall)
        self.log("test_F1_micro", f1)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """
        FROM PYTORCH LIGHTNING DOCUMENTATION

        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.hparams.adafactor else AdamW
        if self.hparams.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.hparams.adam_beta1, self.hparams.adam_beta2),
                "eps": self.hparams.adam_epsilon,
            }

        optimizer_kwargs["lr"] = self.hparams.learning_rate

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = self._get_lr_scheduler(self.hparams.max_steps, optimizer)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def _get_lr_scheduler(self, num_training_steps, optimizer):
        schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == "constant":
            scheduler = schedule_func(optimizer)
        elif self.hparams.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(
                optimizer, num_warmup_steps=self.hparams.warmup_steps
            )
        elif self.hparams.lr_scheduler == "inverse_square_root":
            # args = {"warmup_updates": self.hparams.warmup_steps, "lr": [self.hparams.learning_rate]}
            scheduler = schedule_func(
                optimizer, num_warmup_steps=self.hparams.warmup_steps
            )
        else:
            scheduler = schedule_func(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=num_training_steps,
            )
        return scheduler

    def compute_metrics(self, preds, labels):
        metric_name = (
            "rouge"  # if self.hparams.task.startswith("summarization") else "sacrebleu"
        )
        metric = load_metric(metric_name)

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            if metric_name == "rouge":
                preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
                labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
            else:  # sacrebleu
                labels = [[label] for label in labels]

            return preds, labels

        # preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.hparams.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        if metric_name == "rouge":
            result = metric.compute(
                predictions=decoded_preds, references=decoded_labels, use_stemmer=True
            )
            # Extract a few results from ROUGE
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        else:
            result = metric.compute(
                predictions=decoded_preds, references=decoded_labels
            )
            result = {"bleu": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
