# Results

This document presents the detailed results of entity and relation extraction models. It includes per class entity and relation F1 scores, both micro and macro, alongside other key evaluation metrics. These results are crucial in understanding the performance of the models in identifying entities and relationships within a given text corpus. The results are obtained from the test set portion of the fine-grained expert-annotated corpus for fair comparison consisting of 108 randomly sampled texts.

## Evaluation Metrics Explained

To assess the performance of our models, we have employed several metrics. These are crucial for understanding the effectiveness of the entity recognition and relation extraction processes.

### Named Entity Recognition (NER)

- **Definition**: Named Entity Recognition is the process of identifying and classifying key information (entities) in text into predefined categories.
- **Criteria for Correct Identification**:
  - An entity is considered correctly identified if both the entity type (e.g., person, organization) and its span (the exact start and end points in the text) are predicted accurately.

### Relation Extraction (RE)

Relation Extraction is classified into two types based on the strictness of the criteria:

1. **Strict Relation Extraction**:
   - **Criteria**: A relation is considered correct if the model correctly identifies the type of relation and the spans of the two related entities. The entity type is not considered in this evaluation.
2. **Loose Relation Extraction**:
   - **Criteria**: A relation is deemed correct if the relation type and the two related entities are correctly identified, both in terms of their span and entity type.

### Core Metrics

The performance of the models is evaluated using the following metrics:

- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.
- **Precision**: The proportion of correctly identified positive cases among all cases that the model identified as positive.
- **Recall**: The proportion of actual positive cases that the model correctly identified.
- **Support**: The number of actual occurrences of the class in the specified dataset.



## Detailed Results

### SpERT (Token Classification)

#### Fine-Grained

##### Entity Classes: 1

NER

```
    type    precision       recall     f1-score      support
  Entity        86.63        93.37        89.88          347

   micro        86.63        93.37        89.88          347
   macro        86.63        93.37        89.88          347
```

Strict RE 

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        30.43        70.00        42.42           20
  hasPatient        81.12        84.67        82.86          137
 hasProperty        83.33       100.00        90.91            5
    hasAgent        71.43        62.50        66.67            8
     hasPart        60.56        70.49        65.15           61

       micro        67.39        79.49        72.94          234
       macro        71.15        81.28        74.67          234
```

Loose RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        30.43        70.00        42.42           20
  hasPatient        81.12        84.67        82.86          137
 hasProperty        83.33       100.00        90.91            5
    hasAgent        71.43        62.50        66.67            8
     hasPart        60.56        70.49        65.15           61

       micro        67.39        79.49        72.94          234
       macro        71.15        81.28        74.67          234
```



##### Entity Classes: 5

NER

```
            type    precision       recall     f1-score      support
  PhysicalObject        79.02        86.76        82.71          204
        Activity        97.98        95.10        96.52          102
           State        84.85        96.55        90.32           29
         Process       100.00        83.33        90.91            6
        Property        83.33        83.33        83.33            6

           micro        85.01        89.91        87.39          347
           macro        89.04        89.02        88.76          347
```

Strict RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        26.19        55.00        35.48           20
  hasPatient        76.81        77.37        77.09          137
 hasProperty        33.33        20.00        25.00            5
    hasAgent        75.00        75.00        75.00            8
     hasPart        58.21        63.93        60.94           61

       micro        63.60        70.94        67.07          234
       macro        61.59        65.22        62.25          234
```

Loose RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        26.19        55.00        35.48           20
  hasPatient        78.99        79.56        79.27          137
 hasProperty        33.33        20.00        25.00            5
    hasAgent        75.00        75.00        75.00            8
     hasPart        58.21        63.93        60.94           61

       micro        64.75        72.22        68.28          234
       macro        61.95        65.58        62.62          234
```



##### Entity Classes: 32

NER

```
                type    precision       recall     f1-score      support
Property/UndesirableProperty       100.00       100.00       100.00            5
Activity/SupportingActivity        80.00        75.00        77.42           16
Activity/MaintenanceActivity        96.55        97.67        97.11           86
PhysicalObject/StoringObject        78.57        68.75        73.33           16
PhysicalObject/HoldingObject        33.33        42.11        37.21           19
PhysicalObject/Substance       100.00       100.00       100.00            2
PhysicalObject/RestrictingObject       100.00        25.00        40.00            8
PhysicalObject/InterfacingObject        50.00        16.67        25.00            6
PhysicalObject/PresentingObject        33.33        15.38        21.05           13
PhysicalObject/ProtectingObject        53.33        72.73        61.54           11
PhysicalObject/MatterProcessingObject        76.47        54.17        63.41           24
State/UndesirableState        82.35        96.55        88.89           29
Process/UndesirableProcess       100.00        83.33        90.91            6
PhysicalObject/DrivingObject        71.43        62.50        66.67            8
PhysicalObject/GuidingObject        63.16        80.00        70.59           15
PhysicalObject/InformationProcessingObject         0.00         0.00         0.00            5
PhysicalObject/Organism       100.00        50.00        66.67            2
PhysicalObject/EmittingObject        66.67        72.73        69.57           11
PhysicalObject/ControllingObject        31.25        38.46        34.48           13
PhysicalObject/SensingObject        66.67        60.00        63.16           10
            Property         0.00         0.00         0.00            1
PhysicalObject/GeneratingObject        47.06       100.00        64.00            8
      PhysicalObject        33.33        50.00        40.00            2
PhysicalObject/TransformingObject        85.71        52.17        64.86           23
PhysicalObject/CoveringObject        66.67        50.00        57.14            8

               micro        73.19        70.03        71.58          347
               macro        64.64        58.53        58.92          347
```

Strict RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        22.22        30.00        25.53           20
  hasPatient        41.27        37.96        39.54          137
 hasProperty        20.00        20.00        20.00            5
    hasAgent        42.86        37.50        40.00            8
     hasPart        33.96        29.51        31.58           61

       micro        37.56        35.47        36.48          234
       macro        43.39        42.49        42.78          234
```

Loose RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        44.44        60.00        51.06           20
  hasPatient        77.78        71.53        74.52          137
 hasProperty        40.00        40.00        40.00            5
    hasAgent        57.14        50.00        53.33            8
     hasPart        56.60        49.18        52.63           61

       micro        67.42        63.68        65.49          234
       macro        62.66        61.79        61.93          234
```



##### Entity Classes: 224

NER

```
                type    precision       recall     f1-score      support
HighVoltageConnectingObject         0.00         0.00         0.00            2
SignalConvertingObject        60.00        50.00        54.55            6
      JointingObject       100.00       100.00       100.00            1
       ClosureObject       100.00        66.67        80.00            3
              Modify         0.00         0.00         0.00            2
MatterReshapingObject         0.00         0.00         0.00            4
EnclosedStationaryStoringObject        75.00       100.00        85.71            9
              Liquid       100.00       100.00       100.00            1
            Diagnose         0.00         0.00         0.00            1
MechanicalToElectricalEnergyGeneratingObject       100.00       100.00       100.00            1
ReturnFlowRestrictingObject         0.00         0.00         0.00            1
              Person       100.00        50.00        66.67            2
ElectromagneticRotationalDrivingObject         0.00         0.00         0.00            1
    AssemblingObject         0.00         0.00         0.00            1
 WirelessPowerObject         0.00         0.00         0.00            1
OpenEnclosureGuidingObject       100.00       100.00       100.00            1
      PhysicalObject        50.00        50.00        50.00            2
VisibleStateIndicator         0.00         0.00         0.00            6
              Adjust         0.00         0.00         0.00            1
       DegradedState        50.00        11.11        18.18            9
MechanicalEnergyTransformingObject       100.00       100.00       100.00            4
ForceSeparatingObject       100.00         7.69        14.29           13
              Repair        94.44        94.44        94.44           36
ElectricCoolingObject       100.00       100.00       100.00            4
ThermalCoolingObject       100.00       100.00       100.00            1
ElectricEnergyTransformingObject       100.00        40.00        57.14            5
ElectricityStabilisingObject         0.00         0.00         0.00            2
WearProtectingObject        57.14        80.00        66.67            5
ElectromagneticLinearDrivingObject         0.00         0.00         0.00            1
   ReinforcingObject         0.00         0.00         0.00            5
ChemicalToElectricalEnergyGeneratingObject         0.00         0.00         0.00            0
FluidSignallingObject         0.00         0.00         0.00            2
           Calibrate       100.00       100.00       100.00            1
ElectricityRestrictingObject       100.00        40.00        57.14            5
LowVoltageConnectingObject         0.00         0.00         0.00            2
             Replace        95.65       100.00        97.78           44
     InfillingObject        50.00        50.00        50.00            2
ElectricSignalRelayingObject         0.00         0.00         0.00            1
MatterProcessingObject         0.00         0.00         0.00            0
ElectricControllingObject         0.00         0.00         0.00            2
OvercurrentProtectingObject       100.00       100.00       100.00            2
ElectricEarthingObject         0.00         0.00         0.00            4
MechanicalEnergyGuidingObject        66.67       100.00        80.00            2
MoveableStoringObject       100.00        50.00        66.67            2
      CarryingObject         0.00         0.00         0.00            3
ThermalEnergyTransferObject         0.00         0.00         0.00            1
    UndesirableState        42.86        75.00        54.55            4
ElectricSignalProcessingObject         0.00         0.00         0.00            2
OpenFlowControllingObject         0.00         0.00         0.00            3
ElectricSignalGuidingObject         0.00         0.00         0.00            1
LiquidFlowGeneratingObject        75.00       100.00        85.71            3
SealedFluidVaryingObject         0.00         0.00         0.00            0
MechanicalEnergyDrivingObject         0.00         0.00         0.00            1
SealedFluidSwitchingObject         0.00         0.00         0.00            0
GaseousFlowGeneratingObject       100.00       100.00       100.00            4
  ForceSensingObject         0.00         0.00         0.00            1
ElectricEnergyGuidingObject        62.50        83.33        71.43            6
             Service        25.00        50.00        33.33            2
             Operate         0.00         0.00         0.00            3
ThermalEnergyGuidingObject         0.00         0.00         0.00            3
     FinishingObject       100.00        33.33        50.00            3
   SpaceAccessObject        33.33       100.00        50.00            2
StructuralSupportingObject        12.50        50.00        20.00            4
               Admin       100.00       100.00       100.00            1
MagneticForceDrivingObject         0.00         0.00         0.00            1
            Property         0.00         0.00         0.00            1
   PositioningObject         0.00         0.00         0.00            0
  SpaceLinkingObject         0.00         0.00         0.00            1
             Inspect       100.00       100.00       100.00            1
HandInteractionObject         0.00         0.00         0.00            0
ElectrochemicalStoringObject         0.00         0.00         0.00            4
     FasteningObject         0.00         0.00         0.00            0
PotentialConnectingObject       100.00       100.00       100.00            1
 UndesirableProperty       100.00       100.00       100.00            5
MechanicalSeparatingObject        63.64        87.50        73.68            8
            Assemble         0.00         0.00         0.00            3
  UndesirableProcess       100.00        83.33        90.91            6
PressureProtectingObject       100.00        50.00        66.67            4
             Measure         0.00         0.00         0.00            3
         LightObject        66.67       100.00        80.00            2
       FramingObject         0.00         0.00         0.00            3
             Perform         0.00         0.00         0.00            1
AcousticWaveEmittingObject         0.00         0.00         0.00            2
  PowerSensingObject         0.00         0.00         0.00            4
    CombustionEngine       100.00       100.00       100.00            4
ElectricSeparatingObject         0.00         0.00         0.00            1
ConcentrationSensingObject       100.00        60.00        75.00            5
ElectricEnergyConvertingObject         0.00         0.00         0.00            4
                 Gas       100.00       100.00       100.00            1
SurfaceTreatmentObject         0.00         0.00         0.00            1
MultipleFormPresentingObject         0.00         0.00         0.00            6
MechanicalMovementControllingObject       100.00       100.00       100.00            1
InductiveStoringObject         0.00         0.00         0.00            1
         FailedState        65.22        93.75        76.92           16
ChemicalSeparatingObject       100.00       100.00       100.00            1
       ScalarDisplay         0.00         0.00         0.00            1
                Move       100.00       100.00       100.00            3
ClosedEnclosureGuidingObject        66.67       100.00        80.00            2
     EnclosingObject       100.00       100.00       100.00            3

               micro        72.14        58.21        64.43          347
               macro        41.54        39.96        38.86          347
```

Strict RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        31.25        25.00        27.78           20
  hasPatient        35.37        21.17        26.48          137
 hasProperty         0.00         0.00         0.00            5
    hasAgent        33.33        25.00        28.57            8
     hasPart        34.15        22.95        27.45           61

       micro        34.42        22.65        27.32          234
       macro        39.02        32.35        35.05          234
```

Loose RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        50.00        40.00        44.44           20
  hasPatient        81.71        48.91        61.19          137
 hasProperty        66.67        80.00        72.73            5
    hasAgent        50.00        37.50        42.86            8
     hasPart        56.10        37.70        45.10           61

       micro        70.13        46.15        55.67          234
       macro        67.41        57.35        61.05          234
```



### Coarse-Grained → Fine-Grained

##### Entity Classes: 1

NER

```
    type    precision       recall     f1-score      support
  Entity        86.40        93.37        89.75          347

   micro        86.40        93.37        89.75          347
   macro        86.40        93.37        89.75          347
```

Strict RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        34.78        80.00        48.48           20
  hasPatient        78.23        83.94        80.99          137
 hasProperty        66.67        80.00        72.73            5
    hasAgent        87.50        87.50        87.50            8
     hasPart        64.29        73.77        68.70           61

       micro        67.86        81.20        73.93          234
       macro        71.91        84.20        76.40          234
```

Loose RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        34.78        80.00        48.48           20
  hasPatient        78.23        83.94        80.99          137
 hasProperty        66.67        80.00        72.73            5
    hasAgent        87.50        87.50        87.50            8
     hasPart        64.29        73.77        68.70           61

       micro        67.86        81.20        73.93          234
       macro        71.91        84.20        76.40          234
```



##### Entity Classes: 5

NER

```
            type    precision       recall     f1-score      support
  PhysicalObject        79.91        89.71        84.53          204
        Activity        98.06        99.02        98.54          102
           State        90.32        96.55        93.33           29
         Process       100.00       100.00       100.00            6
        Property        83.33        83.33        83.33            6

           micro        86.13        93.08        89.47          347
           macro        90.33        93.72        91.95          347
```

Strict RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        27.45        70.00        39.44           20
  hasPatient        77.63        86.13        81.66          137
 hasProperty        83.33       100.00        90.91            5
    hasAgent        75.00        75.00        75.00            8
     hasPart        61.97        72.13        66.67           61

       micro        65.29        81.20        72.38          234
       macro        70.90        83.88        75.61          234
```

Loose RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        27.45        70.00        39.44           20
  hasPatient        78.29        86.86        82.35          137
 hasProperty        83.33       100.00        90.91            5
    hasAgent        75.00        75.00        75.00            8
     hasPart        61.97        72.13        66.67           61

       micro        65.64        81.62        72.76          234
       macro        71.01        84.00        75.73          234
```



##### Entity Classes: 32

NER

```
                type    precision       recall     f1-score      support
Property/UndesirableProperty       100.00       100.00       100.00            5
Activity/SupportingActivity        76.92        62.50        68.97           16
Activity/MaintenanceActivity        94.38        97.67        96.00           86
PhysicalObject/StoringObject        57.89        68.75        62.86           16
PhysicalObject/HoldingObject        43.24        84.21        57.14           19
PhysicalObject/Substance       100.00        50.00        66.67            2
PhysicalObject/RestrictingObject         0.00         0.00         0.00            8
PhysicalObject/InterfacingObject        50.00        16.67        25.00            6
PhysicalObject/HumanInteractionObject         0.00         0.00         0.00            0
PhysicalObject/PresentingObject        50.00         7.69        13.33           13
PhysicalObject/ProtectingObject        37.50        54.55        44.44           11
PhysicalObject/MatterProcessingObject        63.16        50.00        55.81           24
State/UndesirableState        87.50        96.55        91.80           29
Process/UndesirableProcess       100.00        83.33        90.91            6
PhysicalObject/DrivingObject       100.00        62.50        76.92            8
PhysicalObject/GuidingObject        47.83        73.33        57.89           15
PhysicalObject/InformationProcessingObject         0.00         0.00         0.00            5
PhysicalObject/Organism        50.00        50.00        50.00            2
PhysicalObject/EmittingObject        72.73        72.73        72.73           11
PhysicalObject/ControllingObject        21.05        30.77        25.00           13
PhysicalObject/SensingObject        37.50        60.00        46.15           10
            Property         0.00         0.00         0.00            1
PhysicalObject/GeneratingObject        40.00       100.00        57.14            8
      PhysicalObject        50.00        50.00        50.00            2
PhysicalObject/TransformingObject        80.00        34.78        48.48           23
PhysicalObject/CoveringObject        71.43        62.50        66.67            8

               micro        66.02        68.30        67.14          347
               macro        55.04        52.64        50.92          347
```

Strict RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        15.38        30.00        20.34           20
  hasPatient        32.41        34.31        33.33          137
 hasProperty        66.67        80.00        72.73            5
    hasAgent        28.57        25.00        26.67            8
     hasPart        23.19        26.23        24.62           61

       micro        29.00        33.33        31.01          234
       macro        44.37        49.26        46.28          234
```

Loose RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        30.77        60.00        40.68           20
  hasPatient        78.62        83.21        80.85          137
 hasProperty        83.33       100.00        90.91            5
    hasAgent        57.14        50.00        53.33            8
     hasPart        60.87        68.85        64.62           61

       micro        66.91        76.92        71.57          234
       macro        68.46        77.01        71.73          234
```

##### Entity Classes: 224

NER

```
                type    precision       recall     f1-score      support
PhysicalObject/ControllingObject/ElectricEarthingObject         0.00         0.00         0.00            4
PhysicalObject/ProtectingObject/FireProtectingObject         0.00         0.00         0.00            0
PhysicalObject/HoldingObject/StructuralSupportingObject        12.50        50.00        20.00            4
Activity/SupportingActivity/Admin         0.00         0.00         0.00            1
State/UndesirableState/DegradedState         0.00         0.00         0.00            9
PhysicalObject/ProtectingObject/WearProtectingObject        50.00        80.00        61.54            5
PhysicalObject/InterfacingObject/LowVoltageConnectingObject         0.00         0.00         0.00            2
PhysicalObject/Organism/Person       100.00        50.00        66.67            2
PhysicalObject/MatterProcessingObject/AssemblingObject         0.00         0.00         0.00            1
PhysicalObject/ControllingObject/ElectricSeparatingObject         0.00         0.00         0.00            1
PhysicalObject/MatterProcessingObject/ForceSeparatingObject         0.00         0.00         0.00           13
Activity/MaintenanceActivity/Calibrate        50.00       100.00        66.67            1
PhysicalObject/ControllingObject/MechanicalMovementControllingObject       100.00       100.00       100.00            1
PhysicalObject/CoveringObject/FinishingObject       100.00        33.33        50.00            3
PhysicalObject/SensingObject         0.00         0.00         0.00            0
            Property         0.00         0.00         0.00            1
PhysicalObject/SensingObject/TimeRatingObject         0.00         0.00         0.00            0
PhysicalObject/StoringObject/MoveableStoringObject        50.00        50.00        50.00            2
PhysicalObject/StoringObject/EnclosedStationaryStoringObject        69.23       100.00        81.82            9
PhysicalObject/ProtectingObject/OvercurrentProtectingObject         0.00         0.00         0.00            2
PhysicalObject/PresentingObject/MultipleFormPresentingObject         0.00         0.00         0.00            6
Property/UndesirableProperty       100.00       100.00       100.00            5
PhysicalObject/TransformingObject/SignalConvertingObject        66.67        33.33        44.44            6
PhysicalObject/EmittingObject/ThermalCoolingObject        33.33       100.00        50.00            1
PhysicalObject/PresentingObject/VisibleStateIndicator         0.00         0.00         0.00            6
PhysicalObject/ControllingObject/SpaceAccessObject        25.00       100.00        40.00            2
PhysicalObject/EmittingObject/WirelessPowerObject         0.00         0.00         0.00            1
PhysicalObject/GuidingObject/ClosedEnclosureGuidingObject        25.00       100.00        40.00            2
PhysicalObject/RestrictingObject/ElectricityStabilisingObject         0.00         0.00         0.00            2
PhysicalObject/InformationProcessingObject/ElectricSignalProcessingObject         0.00         0.00         0.00            2
State/UndesirableState        37.50        75.00        50.00            4
PhysicalObject/DrivingObject/CombustionEngine       100.00       100.00       100.00            4
PhysicalObject/InterfacingObject/HighVoltageConnectingObject         0.00         0.00         0.00            2
Process/UndesirableProcess       100.00       100.00       100.00            6
PhysicalObject/TransformingObject/ElectricEnergyTransformingObject         0.00         0.00         0.00            5
PhysicalObject/DrivingObject/ElectromagneticRotationalDrivingObject         0.00         0.00         0.00            1
PhysicalObject/HoldingObject/FasteningObject         0.00         0.00         0.00            0
PhysicalObject/ControllingObject/SealedFluidVaryingObject         0.00         0.00         0.00            0
PhysicalObject/Substance/Liquid       100.00       100.00       100.00            1
PhysicalObject/InformationProcessingObject/FluidSignallingObject         0.00         0.00         0.00            2
PhysicalObject/GeneratingObject/LiquidFlowGeneratingObject        42.86       100.00        60.00            3
PhysicalObject/TransformingObject/MatterReshapingObject         0.00         0.00         0.00            4
PhysicalObject/GuidingObject/OpenEnclosureGuidingObject         0.00         0.00         0.00            1
PhysicalObject/DrivingObject/MechanicalEnergyDrivingObject         0.00         0.00         0.00            1
PhysicalObject/EmittingObject/ThermalEnergyTransferObject         0.00         0.00         0.00            1
PhysicalObject/HoldingObject/ReinforcingObject         0.00         0.00         0.00            5
PhysicalObject/SensingObject/TemperatureSensingObject         0.00         0.00         0.00            0
PhysicalObject/HoldingObject/PositioningObject         0.00         0.00         0.00            0
      PhysicalObject        14.29        50.00        22.22            2
PhysicalObject/GeneratingObject/MechanicalToElectricalEnergyGeneratingObject       100.00       100.00       100.00            1
Activity/SupportingActivity/Operate         0.00         0.00         0.00            3
PhysicalObject/SensingObject/PressureSensingObject         0.00         0.00         0.00            0
PhysicalObject/RestrictingObject/MovementRestrictingObject         0.00         0.00         0.00            0
PhysicalObject/EmittingObject/AcousticWaveEmittingObject         0.00         0.00         0.00            2
PhysicalObject/ProtectingObject/TemperatureProtectingObject         0.00         0.00         0.00            0
PhysicalObject/TransformingObject/MechanicalEnergyTransformingObject       100.00       100.00       100.00            4
PhysicalObject/GuidingObject/ThermalEnergyGuidingObject         0.00         0.00         0.00            3
PhysicalObject/ProtectingObject/PressureProtectingObject         0.00         0.00         0.00            4
PhysicalObject/SensingObject/ConcentrationSensingObject         0.00         0.00         0.00            5
Activity/MaintenanceActivity/Adjust         0.00         0.00         0.00            1
PhysicalObject/EmittingObject/LightObject        33.33       100.00        50.00            2
Activity/MaintenanceActivity/Inspect       100.00       100.00       100.00            1
PhysicalObject/HoldingObject/CarryingObject       100.00        33.33        50.00            3
PhysicalObject/PresentingObject/GraphicalDisplay         0.00         0.00         0.00            0
PhysicalObject/PresentingObject/ScalarDisplay         0.00         0.00         0.00            1
PhysicalObject/SensingObject/LevelSensingObject         0.00         0.00         0.00            0
Activity/SupportingActivity/Modify         0.00         0.00         0.00            2
PhysicalObject/Substance/Gas       100.00       100.00       100.00            1
PhysicalObject/ControllingObject/ElectricControllingObject         0.00         0.00         0.00            2
PhysicalObject/GeneratingObject/GaseousFlowGeneratingObject        60.00        75.00        66.67            4
PhysicalObject/ControllingObject/OpenFlowControllingObject         0.00         0.00         0.00            3
PhysicalObject/SensingObject/PowerSensingObject         0.00         0.00         0.00            4
PhysicalObject/TransformingObject/ElectricEnergyConvertingObject         0.00         0.00         0.00            4
Activity/MaintenanceActivity/Service        25.00        50.00        33.33            2
PhysicalObject/MatterProcessingObject/MechanicalSeparatingObject        42.86        75.00        54.55            8
Activity/MaintenanceActivity/Repair       100.00        97.22        98.59           36
PhysicalObject/GeneratingObject/ChemicalToElectricalEnergyGeneratingObject         0.00         0.00         0.00            0
PhysicalObject/GuidingObject/MechanicalEnergyGuidingObject        33.33       100.00        50.00            2
PhysicalObject/InterfacingObject/PotentialConnectingObject       100.00       100.00       100.00            1
Activity/MaintenanceActivity/Replace        89.80       100.00        94.62           44
Activity/SupportingActivity/Assemble         0.00         0.00         0.00            3
PhysicalObject/SensingObject/ForceSensingObject         0.00         0.00         0.00            1
PhysicalObject/MatterProcessingObject/GrindingAndCrushingObject         0.00         0.00         0.00            0
PhysicalObject/GuidingObject/ElectricEnergyGuidingObject        57.14        66.67        61.54            6
PhysicalObject/ControllingObject/SealedFluidSwitchingObject         0.00         0.00         0.00            0
PhysicalObject/EmittingObject/ElectricCoolingObject        57.14       100.00        72.73            4
PhysicalObject/RestrictingObject/ElectricityRestrictingObject         0.00         0.00         0.00            5
PhysicalObject/CoveringObject/ClosureObject        50.00        66.67        57.14            3
PhysicalObject/CoveringObject/InfillingObject       100.00        50.00        66.67            2
PhysicalObject/HoldingObject/JointingObject        50.00       100.00        66.67            1
PhysicalObject/GuidingObject/ElectricSignalGuidingObject         0.00         0.00         0.00            1
PhysicalObject/DrivingObject/ElectromagneticLinearDrivingObject         0.00         0.00         0.00            1
Activity/SupportingActivity/Measure         0.00         0.00         0.00            3
PhysicalObject/StoringObject/InductiveStoringObject         0.00         0.00         0.00            1
PhysicalObject/HoldingObject/EnclosingObject       100.00       100.00       100.00            3
PhysicalObject/MatterProcessingObject/ChemicalSeparatingObject         0.00         0.00         0.00            1
PhysicalObject/DrivingObject/MagneticForceDrivingObject         0.00         0.00         0.00            1
Activity/MaintenanceActivity/Diagnose        50.00       100.00        66.67            1
Activity/SupportingActivity/Perform         0.00         0.00         0.00            1
PhysicalObject/StoringObject/ElectrochemicalStoringObject       100.00        25.00        40.00            4
PhysicalObject/InformationProcessingObject/ElectricSignalRelayingObject         0.00         0.00         0.00            1
Activity/SupportingActivity/Move       100.00        66.67        80.00            3
State/UndesirableState/FailedState        66.67       100.00        80.00           16
PhysicalObject/MatterProcessingObject/SurfaceTreatmentObject         0.00         0.00         0.00            1
PhysicalObject/RestrictingObject/ReturnFlowRestrictingObject         0.00         0.00         0.00            1
PhysicalObject/InterfacingObject/SpaceLinkingObject         0.00         0.00         0.00            1
PhysicalObject/HoldingObject/FramingObject         0.00         0.00         0.00            3

               micro        57.01        53.89        55.41          347
               macro        27.02        32.03        27.03          347
```

Strict RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        13.79        20.00        16.33           20
  hasPatient        15.45        13.87        14.62          137
 hasProperty         0.00         0.00         0.00            5
    hasAgent        14.29        12.50        13.33            8
     hasPart        16.00        13.11        14.41           61

       micro        16.13        14.96        15.52          234
       macro        26.59        26.58        26.45          234
```

Loose RE

```
        type    precision       recall     f1-score      support
    contains       100.00       100.00       100.00            3
         isA        37.93        55.00        44.90           20
  hasPatient        82.11        73.72        77.69          137
 hasProperty        80.00        80.00        80.00            5
    hasAgent        71.43        62.50        66.67            8
     hasPart        74.00        60.66        66.67           61

       micro        74.19        68.80        71.40          234
       macro        74.25        71.98        72.65          234
```

### REBEL (Sequence-to-Sequence)

#### Fine-Grained

##### Entity Classes: 2

Strict RE

```
        type    precision       recall     f1-score      support
        is a        23.53        40.00        29.63           20
 has patient        78.52        77.37        77.94          137
   has agent        83.33        62.50        71.43            8
    contains       100.00       100.00       100.00            3
    has part        63.93        63.93        63.93           61
has property       100.00        60.00        75.00            5

       micro        67.77        70.09        68.91          234
       macro        74.89        67.30        69.66          234
```

Loose RE

```
        type    precision       recall     f1-score      support
        is a        23.53        40.00        29.63           20
 has patient        78.52        77.37        77.94          137
   has agent        83.33        62.50        71.43            8
    contains       100.00       100.00       100.00            3
    has part        63.93        63.93        63.93           61
has property       100.00        60.00        75.00            5

       micro        67.77        70.09        68.91          234
       macro        74.89        67.30        69.66          234
```

##### Entity Classes: 5

Strict RE

```
        type    precision       recall     f1-score      support
        is a        22.73        50.00        31.25           20
 has patient        77.86        79.56        78.70          137
   has agent        83.33        62.50        71.43            8
    contains        60.00       100.00        75.00            3
    has part        63.08        67.21        65.08           61
has property        25.00        20.00        22.22            5

       micro        64.02        72.22        67.87          234
       macro        55.33        63.21        57.28          234
```

Loose RE

```
        type    precision       recall     f1-score      support
        is a        22.73        50.00        31.25           20
 has patient        78.57        80.29        79.42          137
   has agent        83.33        62.50        71.43            8
    contains        60.00       100.00        75.00            3
    has part        63.08        67.21        65.08           61
has property        25.00        20.00        22.22            5

       micro        64.39        72.65        68.27          234
       macro        55.45        63.33        57.40          234
```

##### Entity Classes: 32

Strict RE

```
        type    precision       recall     f1-score      support
        is a         0.00         0.00         0.00           20
 has patient         0.00         0.00         0.00          137
   has agent         0.00         0.00         0.00            8
    contains         0.00         0.00         0.00            3
    has part         3.45         3.28         3.36           61
has property         0.00         0.00         0.00            5

       micro         0.79         0.85         0.82          234
       macro         0.57         0.55         0.56          234
```

Loose RE

```
        type    precision       recall     f1-score      support
        is a        16.28        35.00        22.22           20
 has patient        78.26        78.83        78.55          137
   has agent        83.33        62.50        71.43            8
    contains       100.00       100.00       100.00            3
    has part        70.69        67.21        68.91           61
has property        80.00        80.00        80.00            5

       micro        66.40        71.79        68.99          234
       macro        71.43        70.59        70.18          234
```

##### Entity Classes: 224

Strict RE 

```
        type    precision       recall     f1-score      support
        is a         0.00         0.00         0.00         20.0
 has patient         0.00         0.00         0.00        137.0
   has agent         0.00         0.00         0.00          8.0
    contains         0.00         0.00         0.00          3.0
    has part         0.00         0.00         0.00         61.0
has property         0.00         0.00         0.00          5.0

       micro         0.00         0.00         0.00        234.0
       macro         0.00         0.00         0.00        234.0
```

Loose RE

```
        type    precision       recall     f1-score      support
        is a        21.95        45.00        29.51           20
 has patient        77.54        78.10        77.82          137
   has agent        60.00        37.50        46.15            8
    contains       100.00       100.00       100.00            3
    has part        68.00        55.74        61.26           61
has property        60.00        60.00        60.00            5

       micro        65.43        67.95        66.67          234
       macro        55.36        53.76        53.53          234
```

### Coarse-Grained → Fine-Grained

##### Entity Classes: 2

Strict RE

```
        type    precision       recall     f1-score      support
        is a        25.53        60.00        35.82           20
 has patient        81.56        83.94        82.73          137
   has agent        83.33        62.50        71.43            8
    contains       100.00       100.00       100.00            3
    has part        76.36        68.85        72.41           61
has property        80.00        80.00        80.00            5

       micro        70.43        77.35        73.73          234
       macro        74.46        75.88        73.73          234
```

Loose RE

```
        type    precision       recall     f1-score      support
        is a        25.53        60.00        35.82           20
 has patient        81.56        83.94        82.73          137
   has agent        83.33        62.50        71.43            8
    contains       100.00       100.00       100.00            3
    has part        76.36        68.85        72.41           61
has property        80.00        80.00        80.00            5

       micro        70.43        77.35        73.73          234
       macro        74.46        75.88        73.73          234
```

##### Entity Classes: 5

Strict RE

```
        type    precision       recall     f1-score      support
        is a        30.95        65.00        41.94           20
 has patient        76.26        77.37        76.81          137
   has agent        55.56        62.50        58.82            8
    contains       100.00       100.00       100.00            3
    has part        66.67        68.85        67.74           61
has property        80.00        80.00        80.00            5

       micro        66.28        73.93        69.90          234
       macro        68.24        75.62        70.89          234
```

Loose RE

```
        type    precision       recall     f1-score      support
        is a        30.95        65.00        41.94           20
 has patient        77.70        78.83        78.26          137
   has agent        55.56        62.50        58.82            8
    contains       100.00       100.00       100.00            3
    has part        66.67        68.85        67.74           61
has property        80.00        80.00        80.00            5

       micro        67.05        74.79        70.71          234
       macro        68.48        75.86        71.13          234
```

##### Entity Classes: 32

Strict RE

```
        type    precision       recall     f1-score      support
        is a        14.29        30.00        19.35           20
 has patient        39.16        40.88        40.00          137
   has agent        57.14        50.00        53.33            8
    contains       100.00       100.00       100.00            3
    has part        34.48        32.79        33.61           61
has property        60.00        60.00        60.00            5

       micro        35.66        39.32        37.40          234
       macro        50.85        52.28        51.05          234
```

Loose RE

```
        type    precision       recall     f1-score      support
        is a        33.33        70.00        45.16           20
 has patient        77.62        81.02        79.29          137
   has agent        71.43        62.50        66.67            8
    contains       100.00       100.00       100.00            3
    has part        65.52        62.30        63.87           61
has property        80.00        80.00        80.00            5

       micro        67.83        74.79        71.14          234
       macro        71.32        75.97        72.50          234
```

##### Entity Classes: 224

Strict RE

```
        type    precision       recall     f1-score      support
        is a         0.00         0.00         0.00         20.0
 has patient         0.00         0.00         0.00        137.0
   has agent         0.00         0.00         0.00          8.0
    contains         0.00         0.00         0.00          3.0
    has part         0.00         0.00         0.00         61.0
has property         0.00         0.00         0.00          5.0

       micro         0.00         0.00         0.00        234.0
       macro         0.00         0.00         0.00        234.0
```

Loose RE

```
        type    precision       recall     f1-score      support
        is a        36.59        75.00        49.18           20
 has patient        81.12        84.67        82.86          137
   has agent        85.71        75.00        80.00            8
    contains       100.00       100.00       100.00            3
    has part        75.00        68.85        71.79           61
has property        80.00        80.00        80.00            5

       micro        72.94        79.49        76.07          234
       macro        76.40        80.59        77.31          234
```

