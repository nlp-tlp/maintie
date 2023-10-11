"""Create dataset splits for SpERT and REBEL models


Notes
-----
- entity end index is not inclusive
- only short relation names are used. 

Example
-------
```json
[
    {
        "tokens": [...],
        "entities": [{"type": "", "start": #, "end": #}],
        "relations": [{"type": "", "head": #, "tail": #}]
    }
]
```
"""

import copy
from collections import Counter
import json
import random
import os

GOLD_CORPUS_PATH = "../data/gold_release.json"
SILVER_CORPUS_PATH = "../data/silver_release.json"
ONTOLOGY_PATH = "../data/scheme.json"
DATA_DIR = "./data"
UNTYPED_ENTITY_CLASS_NAME = "Entity"
SEED = 1337
random.seed(SEED)


# Flatten ontology
def flatten_nested(data):
    """
    Flattens a nested array of dictionaries with potential 'children' keys into a single list of dictionaries.

    Parameters:
    - data (list): A list of dictionaries, where each dictionary can have a key 'children' with a nested list.

    Returns:
    - list: A flattened list of dictionaries without the nested 'children' structures.
    """

    flattened = []

    for item in data:
        # Extract item's children if present, or set to an empty list
        children = item.pop("children", [])

        # Add the current item to the flattened list
        flattened.append(item)

        # If the item had children, flatten them recursively
        if children:
            flattened.extend(flatten_nested(children))

    return flattened


def convert_camelcase_to_separated_format(text: str):
    return (
        "".join([" " + char if char.isupper() else char for char in text])
        .strip()
        .lower()
    )


def convert_to_angled_format(text: str, top_level_only: bool = False) -> str:
    """
    Convert CamelCase format strings like "PhysicalObject/EmittingObject/ElectricCoolingObject"
    into a format like "<electric cooling object>".

    Parameters:
    - text (str): The input CamelCase format string.
    - top_level_only (bool): returns root string only.

    Returns:
    - str: The transformed string in the desired format.

    Examples:
    >>> convert_to_angled_format("PhysicalObject/EmittingObject/ElectricCoolingObject")
    "<electric cooling object>"
    >>> convert_to_angled_format("PhysicalObject/EmittingObject/ElectricCoolingObject", 1)
    "<physical object>"
    """

    # Extract the last segment after splitting by '/'
    segment = text.split("/")[-1]

    if top_level_only:
        segment = text.split("/")[0]

    # Convert CamelCase to space separated format
    converted = convert_camelcase_to_separated_format(text=segment)

    return f"<{converted}>"


def get_substring_by_level(s, level):
    # Split the string based on "/"
    parts = s.split("/")

    # Check if level is within range
    if level < 1:
        raise ValueError("Invalid level!")

    # If specified level is greater than available parts, return the original string
    if level > len(parts):
        return s

    # Otherwise, return the combined string using "/" up to the specified level
    return "/".join(parts[:level])


def main(silver_corpus: bool = False):
    print("CREATING DATASETS WITH NORMALISED INPUTS")
    print(f'USING {"SILVER" if silver_corpus else "GOLD"} CORPUS')

    folder_prefix = ""  # "c2t-"
    dataset_type = "s" if silver_corpus else "g"  # g - gold, s - silver

    with open(SILVER_CORPUS_PATH if silver_corpus else GOLD_CORPUS_PATH, "r") as f:
        data = json.load(f)

    output_data = []
    try:
        for item in data:
            entities = item.pop("entities")
            # Remove root from relation types
            relations = [
                {
                    "type": r["type"].split("/")[1] if "/" in r["type"] else r["type"],
                    "head": r["head"],
                    "tail": r["tail"],
                }
                for r in item.pop("relations")
            ]

            output_data.append(
                {
                    "tokens": item["tokens"],
                    "entities": entities,
                    "relations": relations,
                }
            )
    except Exception as e:
        print(f"EXCEPTION: {e} - {item}")
        raise Exception

    # Shuffle and save the data
    random.shuffle(data)

    # Create untyped dataset (*2t-^-0)
    # For SpERT, a single sentinel token is used in place of all entities, "Entity"
    untyped_dataset = []
    for item in output_data:
        # Convert entity classes into single type
        untyped_dataset.append(
            {
                **item,
                "entities": [
                    {
                        **e,
                        "type": UNTYPED_ENTITY_CLASS_NAME,
                    }
                    for e in item["entities"]
                ],
            }
        )

    # Create multi-level datasets (*2t-^-1/2/3)
    multi_level_datasets = {}
    for _level in [1, 2, 3]:
        multi_level_datasets[_level] = []
        for item in output_data:
            # Truncate entity classes based on level...
            multi_level_datasets[_level].append(
                {
                    **item,
                    "entities": [
                        {
                            **e,
                            "type": get_substring_by_level(e["type"], level=_level),
                        }
                        for e in item["entities"]
                    ],
                }
            )

    # Create "_types.json"
    with open(ONTOLOGY_PATH, "r") as f:
        ontology = json.load(f)

    def split_and_save_datasets(data, folder_name: str):
        # Determine the split indices
        train_split_idx = int(0.8 * len(data))
        dev_split_idx = int(0.9 * len(data))

        # Split the data into 80%, 10%, and 10% portions
        train_data = data[:train_split_idx]
        dev_data = data[train_split_idx:dev_split_idx]
        test_data = data[dev_split_idx:]

        datasets = {"train": train_data, "dev": dev_data, "test": test_data}

        # print(f"Train: {len(train_data)} Dev: {len(dev_data)} Test: {len(test_data)}")

        # Check if directory exists. If not, create it.
        if not os.path.exists(folder_name):
            print(f'Creating directory for "{folder_name}"')
            os.makedirs(folder_name)

        for name, _data in datasets.items():
            with open(f"./{folder_name}/maintie_{name}.json", "w") as f:
                json.dump(_data, f, indent=2)

        return datasets

    def create_and_save_entity_type_data(
        folder_name: str, level: int = None, untyped: bool = False
    ):
        _entity_ontology = copy.deepcopy(ontology["entity"])
        _relation_ontology = copy.deepcopy(ontology["relation"])

        entitie_types = {e["fullname"]: e for e in flatten_nested(_entity_ontology)}

        if level:
            unique_entitie_types = {
                get_substring_by_level(e_fullname, level=_level)
                for e_fullname in entitie_types.keys()
            }
            entities = {
                entitie_types[e_fullame]["fullname"]: {
                    "short": entitie_types[e_fullame]["name"],
                    "verbose": entitie_types[e_fullame]["fullname"],
                }
                for e_fullame in unique_entitie_types
            }
            print("unique_entitie_types", len(unique_entitie_types))

        if untyped:
            entities = {
                UNTYPED_ENTITY_CLASS_NAME: {
                    "short": UNTYPED_ENTITY_CLASS_NAME,
                    "verbose": UNTYPED_ENTITY_CLASS_NAME,
                }
            }

        maintie_types = {
            "entities": entities,
            "relations": {
                r["name"]: {
                    "short": r["name"],
                    "verbose": r["fullname"],
                    "symmetric": False,
                }
                for r in flatten_nested(_relation_ontology)
            },
        }

        # Save types
        with open(f"./{folder_name}/maintie_types.json", "w") as f:
            json.dump(maintie_types, f, indent=2)

        # Save object of mapping for REBEL
        with open(f"./{folder_name}/maintie_rebel_mapping.json", "w") as f:
            _entities = {
                k: convert_to_angled_format(v["short"])
                for k, v in maintie_types["entities"].items()
            }
            _relations = {
                k: convert_camelcase_to_separated_format(v["short"])
                for k, v in maintie_types["relations"].items()
            }

            json.dump(
                {
                    "entities": _entities,
                    "rebel_entity_types": list(_entities.values()),
                    "relations": _relations,
                    "rebel_relation_types": list(_relations.values()),
                },
                f,
                indent=2,
            )

    def aggregate_annotations(data):
        """Aggregates entities and relations"""
        all_entities = []  # (ngram, type)
        all_relations = []  # (head, relation, tail)

        for di in data:
            tokens = di["tokens"]
            entities = di["entities"]

            formatted_entities = [
                (" ".join(tokens[e["start"] : e["end"] + 1]), e["type"])
                for e in entities
            ]
            relations = di.get("relations")
            formatted_relations = []
            if relations:
                formatted_relations = [
                    (
                        (
                            " ".join(
                                tokens[
                                    entities[r["head"]]["start"] : entities[r["head"]][
                                        "end"
                                    ]
                                    + 1
                                ]
                            )
                        ),
                        r["type"],
                        (
                            " ".join(
                                tokens[
                                    entities[r["tail"]]["start"] : entities[r["tail"]][
                                        "end"
                                    ]
                                    + 1
                                ]
                            )
                        ),
                    )
                    for r in relations
                ]

            all_entities.extend(formatted_entities)
            all_relations.extend(formatted_relations)

        return all_entities, all_relations

    def get_split_annotation_distributions(data):
        """Returns the distribution of splits"""

        for split, _data in data.items():
            print(split, len(_data))
            # print(_data[0])

            _entities, _relations = aggregate_annotations(_data)

            _entity_counts = Counter(_entities)
            # print(_entity_counts)
            _top_level_entities = [(i[0], i[1].split("/")[0]) for i in _entities]
            # print(_top_level_entities)
            _unique_counts = Counter(
                [j[1] for j in set(i for i in _top_level_entities)]
            )
            _total_counts = Counter([i[1] for i in _top_level_entities])

            print(f"Total: {_total_counts}\nUnique: {_unique_counts}")

            total_counts = Counter([i[1] for i in _relations])
            unique_counts = Counter([i[1] for i in set(_relations)])
            print(f"\nRELATIONS\nTotal: {total_counts}\nUniuqe: {unique_counts}")

            # return counts, unique_counts, total_counts

    # Create untyped dataset
    split_and_save_datasets(
        data=untyped_dataset,
        folder_name=f"./{DATA_DIR}/{folder_prefix}{dataset_type}-0",
    )
    create_and_save_entity_type_data(
        folder_name=f"./{DATA_DIR}/{folder_prefix}{dataset_type}-0", untyped=True
    )

    # Create multi-level datasets
    for _level in [1, 2, 3]:
        # print(f"Processing dataset level: {_level}")

        _dataset = multi_level_datasets[_level]

        _split_datasets = split_and_save_datasets(
            data=_dataset,
            folder_name=f"./{DATA_DIR}/{folder_prefix}{dataset_type}-{_level}",
        )
        create_and_save_entity_type_data(
            folder_name=f"./{DATA_DIR}/{folder_prefix}{dataset_type}-{_level}",
            level=_level,
        )

        # Output metrics on splits
        get_split_annotation_distributions(data=_split_datasets)


if __name__ == "__main__":
    # USE TO CREATE C2T DATASETS
    main()

    # Create silver C2T training dataset
    main(silver_corpus=True)
