# MaintIE Annotation Guidelines

Welcome to the annotation guidelines for our entity recognition and relation extraction task! This document provides detailed instructions to ensure consistent and high-quality annotations.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Entity Annotation](#2-entity-annotation)
   1. [Definition](#21-definition)
   2. [Entity Types](#22-entity-types)
   3. [Annotation Guidelines](#23-annotation-guidelines)
3. [Relation Annotation](#3-relation-annotation)
   1. [Definition](#31-definition)
   2. [Relation Types](#32-relation-types)
   3. [Annotation Guidelines](#33-annotation-guidelines)

## 1. Introduction

This project aims to annotate named entities and their relationships in textual data. Your annotations will contribute to building machine learning models for natural language processing tasks.

### Notation

`(X)[Y]` represents an entity with the text _X_ and class _Y_. `(X1)[Y1]-[R]->(X2)[Y2]` represents a triple composed of two entities and one relation (_R_). Hence, (car)[PhysicalObject]-[hasPart]->(engine)[PhysicalObject] represents the fact that a car of type PhysicalObject has the part engine of type PhysicalObject.

## 2. Entity Annotation

### 2.1. Definition

Entity Annotation is the process of identifying and categorising specific named entities within a text, such as persons, organizations, and locations. In the context of MaintIE, this extends to other categories like "PhysicalObject", "Activity", "Process", "Property", and "State".

Consider the example "replace oil". In this text:

- "replace" is a verb that corresponds to the entity type "Activity".
- "oil" is a noun that corresponds to the entity type "PhysicalObject".

These annotations can be visually represented as:

- `(replace)[Activity]`
- `(oil)[PhysicalObject]`

### 2.2. Entity Types

The five primary entity types, along with their sub-types, are detailed in the [Scheme](./SCHEME/#2-entities). The following sections provide examples illustrating how these entities map syntacically and semantically to MWO short texts.

#### 2.2.1. PhysicalObject

Physical objects are represented as nouns, which can range from single words to multi-word phrases (n-grams).

Examples:

- change out **engine** _[noun]_
- **centrifugal pump** leaking _[noun]_
- **hydraulic cylinder** needs replacement _[noun]_
- clean the **air filter** regularly _[noun]_

#### 2.2.2. State

State expressions are linguistically distinct, often veering away from standard nouns that denote physical objects. These expressions span various constructions, from temporally static verbs to adjectives.

Examples:

- AM/FM aerial bracket **broken_off** _[verb + particle]_
- power **tripped out** _[verb in past tense + particle]_
- faults found on **break_down** _[noun]_
- replace **broken** pump _[adjective]_
- header tank coolant hose **holed** _[past participle]_
- park brake light **staying_on** _[phrasal verb]_

#### 2.2.3. Process

Processes are depicted through various syntactic constructions, most prominently non-finite verb forms.

Examples:

- engine **working** correctly _[present participle]_
- doors **not_opening** properly _[negative present participle]_
- replace air compressor (**bypassing**) _[present participle]_
- air conditioner fan **vibrating** _[present participle]_
- transmission filter light **coming_on** _[phrasal verb]_
- light in cabin **not_working** _[negative present participle]_
- pump is **leaking** _[present participle]_

#### 2.2.4. Activity

Activities in MWO texts are primarily expressed as verbs, gerunds, or participles (both present and past).

Examples:

- **replace** pump impeller _[verb]_
- pump impeller requires **replacing** _[present participle]_
- **cleaned** pump housing _[verb in past tense]_

#### 2.2.5. Property

Properties are generally denoted as nouns.

Examples:

- tank has a **crack** _[noun]_
- alarming high crankcase **pressure** _[noun]_
- front muffler has a **hole** in it _[noun]_
- fill **holes** in bucket _[noun]_
- adjust wheel **alignment** _[noun]_

### 2.3. Annotation Guidelines

#### 2.3.1. Span Selection

Aim to capture the entire entity, omitting any extraneous spaces or punctuation marks.

- Example: In "engine ( leaking )", focus on "leaking" rather than "( leaking )".

When a term or phrase is disrupted by another word, exercise judgement.

- Example: In "blown hose off", only annotate "blown" because the term "blown off" is fragmented by "hose".

Strive to identify entities at their most granular level.

- Example: While "engine oil" can be split into "engine" and "oil", the term "hydraulic cylinder" should be annotated as a whole since "hydraulic" on its own is ambiguous.

#### 2.3.2. Case Sensitivity

Despite the normalisation of the MaintIE corpus for true casing, annotations should be uniform regardless of word capitalisation.

- Example: Both "replace" and "Replace" should be annotated in the same manner.

#### 2.3.3. Ambiguity

Context is key when an entity could be categorised in multiple ways. When faced with ambiguity, opt for a more general category over a specific one. Such annotations, though broad, are valuable as they open avenues for future refinement. This also applies to entities you may be unsure about.

- Example: In the text "replace cable", the exact nature of the "cable" is ambiguous. We can't definitively determine if it's an electrical or structural cable based on the given context. In such cases, it's appropriate to annotate it under a more general Physical Object entity class, ensuring accuracy at a broader level, even if finer details remain unspecified.

#### 2.3.4. Consistency

Ensure consistency across annotations. If you've annotated a particular phrase or term in a certain way in one part of the text, aim to maintain that consistency throughout.

#### 2.3.5. Avoid Over-Annotation

Not every term that stands out is necessarily an entity. It's essential to differentiate between general terms such as stop words and specific entities.

#### 2.3.6. Nested Entities

When annotating nested entities, it's vital to ensure that each individual entity is coherent and unambiguous on its own, even outside the nested context.

- Example: In the text "replace centrifugal pump", both "centrifugal pump" and "pump" can be distinctively identified as entities without confusion. Conversely, in "blown high pressure hose", while "high pressure hose" and "hose" can be treated as clear entities, "pressure hose" should be avoided as it doesn't stand as a commonly recognised term for a physical object.

#### 2.3.7. Feedback and Iteration

Periodically review your annotations and be open to feedback.

## 3. Relation Annotation

### 3.1. Definition

Relation Annotation is the process of identifying and marking relationships between the recognized entities within a text. It aids in establishing the connections between entities, which can be based on actions, ownership, causality, and other relationships. Within the MaintIE framework, this involves determining how entities like "PhysicalObject", "Activity", "Process", "Property", and "State" interrelate.

Take the text "replace broken pump". The named entities consist of:

- `(replace)[Activity/MaintenanceActivity/Replace]`
- `(broken)[State/UndesirableState/FailedState]`
- `(pump)[PhysicalObject/GeneratingObject/LiquidFlowGeneratingObject]`

The relationship between the entities can be visually depicted as:

- `(replace)[Activity/MaintenanceActivity/Replace]-[hasParticipant/hasPatient]->(pump)[PhysicalObject/GeneratingObject/LiquidFlowGeneratingObject]`: the pump is the experiencer of the replace activity.
- `(broken)[State/UndesirableState/FailedState]-[hasParticipant/hasPatient]->(pump)[PhysicalObject/GeneratingObject/LiquidFlowGeneratingObject]`: the pump is the experiencer of the broken state.

### 3.2. Relation Types

The six MaintIE relation types are detailed in the [Scheme](./SCHEME/#3-relations). This section offers examples to illustrate how these relations are applied between named entities identified in MWO short texts.

#### 3.2.1. contains

The `contains` relation typically illustrates a physical object encompassing another, where the latter is often some form of substance.

Example:

- "replace engine oil": `(engine)[...]-[contains]->(oil)[...]`

#### 3.2.2. hasPart

The `hasPart` relation denotes one physical object being a component or part of another.

Example:

- "pump impeller broken": `(pump)[...]-[hasPart]->(impeller)[...]`

#### 3.2.3. hasParticipant/hasPatient

The `hasParticipant/hasPatient` relation connects an activity, state or process to the entity it directly affects, aligning with the PropBank patient role.

Example:

- "pump impeller broken": `(broken)[...]-[hasParticipant/hasPatient]->(impeller)[...]`

#### 3.2.4. hasParticipant/hasAgent

The `hasParticipant/hasAgent` relation ties an activity, state or process to the entity responsible for it, corresponding with the PropBank agent role.

Example:

- "boilermaker to replace pump": `(replace)[...]-[hasParticipant/hasAgent]->(boilermaker)[...]`

#### 3.2.5. hasProperty

The `hasProperty` relation connects physical objects to their inherent or associated properties.

- "check engine pressure": `(engine)[...]-[hasProperty]->(pressure)[...]`

#### 3.2.6. isA

The `isA` relation defines hierarchical relationships or subclassifications between entities.

- "diesel engine overheating": `(diesel engine)[...]-[isA]->(engine)[...]`

### 3.3. Annotation Guidelines

#### 3.3.1. Always Refer to the Context

Before annotating a relationship, always read the entire sentence or relevant context to ensure that the relationship is accurately captured.

#### 3.3.2. Entity Pairs

A relation annotation should always involve a pair of entities. Always ensure that both entities involved in the relation are correctly identified before annotating the relation.

#### 3.3.3. Entity Type Consistency

Ensure that when applying relations that the head and tail entity types make sense. For example, using the hasProperty relation would have the tail as a "property" entity. An incorrect use would be Physical object has property Physical Object, this makes no sense.

#### 3.3.4. Directionality Matters

Relations have a head and a tail. It's essential to keep this direction consistent. For instance, in the `contains` relation, the container is the head, and the contained is the tail.

#### 3.3.5. Avoid Overlapping Relations

Be cautious not to annotate overlapping or redundant relations. If two relations share entities and essentially convey the same information, only one should be chosen based on which provides more context.

#### 3.3.6. Non-Transitivity of Relations

Relations are generally non-transitive. If A has a relation to B, and B has a relation to C, it doesn't necessarily mean A has the same relation to C.

#### 3.3.7. Implicit Relations

Sometimes, relations can be implicit and not explicitly stated in the text. In such cases, use domain knowledge and context to determine the relation, but also be cautious not to infer relations that aren't strongly supported by the text.

#### 3.3.8. PropBank Alignments

To ensure consistent relation annotation, annotators should align relation types, like `hasParticipant/hasAgent`, with corresponding PropBank roles. The [Unified Verb Index (UVI)](https://verbs.colorado.edu/verb-index/index.php) is an instrumental resource for this. While the UVI provides definitions and rolesets for many verbs, it's essential to consider the following scenarios:

1. **Single Role**: Some UVI rolesets, such as "seize up" ([seize_up.02](https://verbs.colorado.edu/propbank/framesets-english-aliases/seize.html)), have only one role. In this example, the role pertains to the entity becoming immobile.

2. **Multiple Roles**: Some rolesets might encompass more than two roles.

3. **Unspecified Roles**: At times, rolesets won't directly state agent or patient. For "leak" ([leak.01](https://verbs.colorado.edu/propbank/framesets-english-aliases/leak.html)), Arg0-DIR and Arg1-PPT are given instead of agent and patient. Here, Arg1-PPT is assumed to be the patient (e.g., the thing experiencing the leak), and Arg0-DIR is treated as the agent (e.g., the entity causing the leak).

4. **Different Semantic Interpretations**: Occasionally, a verb's sense in the UVI may not align with its use in the MWO short text. For instance, "label" ([label.01](https://verbs.colorado.edu/propbank/framesets-english-aliases/label.html)) in the UVI contains roles for both agent and patient, but the "agent" does not represent what is expected in a MWO short text. Instead, "label" might denote the act of attaching a label to an object. Here, the agent is the one performing the labeling, and the patient is the object receiving the label.

If a verb is not in the UVI, or if its roleset doesn't align perfectly with your dataset's context, rely on domain knowledge and sound judgement.
