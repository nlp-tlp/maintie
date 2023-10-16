from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import omegaconf

from train import maintie_general_entities, maintie_level_1_unique_entities

config = AutoConfig.from_pretrained(
    "facebook/bart-large",
    decoder_start_token_id=0,
    early_stopping=False,
    no_repeat_ngram_size=0,
)

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/bart-large",
    use_fast=True,
    additional_special_tokens=[
        "<obj>",
        "<subj>",
        "<triplet>",
        "<head>",
        "</head>",
        "<tail>",
        "</tail>",
        *maintie_general_entities,
        *maintie_level_1_unique_entities,
    ],
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/bart-large",
    config=config,
)
model.resize_token_embeddings(len(tokenizer))

conf = omegaconf.OmegaConf.load("<PATH_TO_MAINTIE_S_1_HYDRA_CONFIG>")
pl_module = BasePLModule(conf, config, tokenizer, model)
model = pl_module.load_from_checkpoint(
    checkpoint_path="<PATH_TO_MAINTIE_S_1_CHECKPOINT>",
    config=config,
    tokenizer=tokenizer,
    model=model,
)

model.model.save_pretrained("../model/maintie")
model.tokenizer.save_pretrained("../model/maintie")
