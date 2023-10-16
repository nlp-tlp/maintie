import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pl_data_modules import BasePLDataModule
from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from generate_samples import GenerateTextSamplesCallback

relations = {
    "no_relation": "no relation",
    "org:alternate_names": "alternate name",
    "org:city_of_branch": "city of headquarters",
    "org:country_of_branch": "country of headquarters",
    "org:dissolved": "dissolved",
    "org:founded_by": "founded by",
    "org:founded": "founded",
    "org:member_of": "member of",
    "org:members": "members",
    "org:number_of_employees/members": "number of members",
    "org:political/religious_affiliation": "affiliation",
    "org:shareholders": "shareholders",
    "org:stateorprovince_of_branch": "state of headquarters",
    "org:top_members/employees": "top members",
    "org:website": "website",
    "per:age": "age",
    "per:cause_of_death": "cause of death",
    "per:charges": "charges",
    "per:children": "children",
    "per:cities_of_residence": "city of residence",
    "per:city_of_birth": "place of birth",
    "per:city_of_death": "place of death",
    "per:countries_of_residence": "country of residence",
    "per:country_of_birth": "country of birth",
    "per:country_of_death": "country of death",
    "per:date_of_birth": "date of birth",
    "per:date_of_death": "date of death",
    "per:employee_of": "employee of",
    "per:identity": "identity",
    "per:origin": "origin",
    "per:other_family": "other family",
    "per:parents": "parents",
    "per:religion": "religion",
    "per:schools_attended": "educated at",
    "per:siblings": "siblings",
    "per:spouse": "spouse",
    "per:stateorprovince_of_birth": "state of birth",
    "per:stateorprovince_of_death": "state of death",
    "per:stateorprovinces_of_residence": "state of residence",
    "per:title": "title",
}


def train(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)

    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id=0,
        early_stopping=False,
        no_repeat_ngram_size=0,
        # cache_dir=conf.cache_dir,
        # revision=conf.model_revision,
        # use_auth_token=True if conf.use_auth_token else None,
    )

    tokenizer_kwargs = {
        "use_fast": conf.use_fast_tokenizer,
        "additional_special_tokens": ["<obj>", "<subj>", "<triplet>"],
    }

    tokenizer = AutoTokenizer.from_pretrained(
        conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
        **tokenizer_kwargs,
    )
    print(f"Base tokenizer size: {len(tokenizer)}")

    if "maintie" in conf.dataset_name.split("/")[-1]:
        print("ADDING SPECIAL TOKENS FOR MAINTIE")
        tokenizer.add_tokens(
            ["<num>", "<id>", "<date>", "<sensitive>"], special_tokens=True
        )

        if "_3" in conf.dataset_name.split("/")[-1]:
            tokenizer.add_tokens(
                [
                    "<physical object>",
                    "<substance>",
                    "<gas>",
                    "<liquid>",
                    "<solid>",
                    "<mixture>",
                    "<organism>",
                    "<person>",
                    "<sensing object>",
                    "<electric potential sensing object>",
                    "<resistivity sensing object>",
                    "<electric current sensing object>",
                    "<density sensing object>",
                    "<field sensing object>",
                    "<flow sensing object>",
                    "<physical dimension sensing object>",
                    "<energy sensing object>",
                    "<power sensing object>",
                    "<time sensing object>",
                    "<level sensing object>",
                    "<humidity sensing object>",
                    "<pressure sensing object>",
                    "<concentration sensing object>",
                    "<radiation sensing object>",
                    "<time rating object>",
                    "<temperature sensing object>",
                    "<multi quantity sensing object>",
                    "<force sensing object>",
                    "<audio visual sensing object>",
                    "<information sensing object>",
                    "<incident sensing object>",
                    "<storing object>",
                    "<capacitive storing object>",
                    "<inductive storing object>",
                    "<electrochemical storing object>",
                    "<information storing object>",
                    "<open stationary storing object>",
                    "<enclosed stationary storing object>",
                    "<moveable storing object>",
                    "<thermal energy storing object>",
                    "<mechanical energy storing object>",
                    "<emitting object>",
                    "<light object>",
                    "<electric heating object>",
                    "<electric cooling object>",
                    "<wireless power object>",
                    "<thermal energy transfer object>",
                    "<combustion heating object>",
                    "<thermal heating object>",
                    "<thermal cooling object>",
                    "<nuclear powered heating object>",
                    "<particle emitting object>",
                    "<acoustic wave emitting object>",
                    "<protecting object>",
                    "<overvoltage protecting object>",
                    "<earth fault current protecting object>",
                    "<overcurrent protecting object>",
                    "<field protecting object>",
                    "<pressure protecting object>",
                    "<fire protecting object>",
                    "<mechanical force protecting object>",
                    "<preventive protecting object>",
                    "<wear protecting object>",
                    "<environment protecting object>",
                    "<temperature protecting object>",
                    "<generating object>",
                    "<mechanical to electrical energy generating object>",
                    "<chemical to electrical energy generating object>",
                    "<solar to electrical energy generating object>",
                    "<signal generating object>",
                    "<continuous transfer object>",
                    "<discontinuous transfer object>",
                    "<liquid flow generating object>",
                    "<gaseous flow generating object>",
                    "<solar to thermal energy generating object>",
                    "<matter processing object>",
                    "<primary forming object>",
                    "<surface treatment object>",
                    "<assembling object>",
                    "<force separating object>",
                    "<thermal separating object>",
                    "<mechanical separating object>",
                    "<electric or magnetic separating object>",
                    "<chemical separating object>",
                    "<grinding and crushing object>",
                    "<agglomerating object>",
                    "<mixing object>",
                    "<reacting object>",
                    "<information processing object>",
                    "<electric signal processing object>",
                    "<electric signal relaying object>",
                    "<optical signalling object>",
                    "<fluid signalling object>",
                    "<mechanical signalling object>",
                    "<multiple kind signalling object>",
                    "<driving object>",
                    "<electromagnetic rotational driving object>",
                    "<electromagnetic linear driving object>",
                    "<magnetic force driving object>",
                    "<piezoelectric driving object>",
                    "<mechanical energy driving object>",
                    "<fluid powered driving object>",
                    "<combustion engine>",
                    "<heat engine>",
                    "<covering object>",
                    "<infilling object>",
                    "<closure object>",
                    "<finishing object>",
                    "<terminating object>",
                    "<hiding object>",
                    "<presenting object>",
                    "<visible state indicator>",
                    "<scalar display>",
                    "<graphical display>",
                    "<acoustic device>",
                    "<tactile device>",
                    "<ornamental object>",
                    "<multiple form presenting object>",
                    "<controlling object>",
                    "<electric controlling object>",
                    "<electric separating object>",
                    "<electric earthing object>",
                    "<sealed fluid switching object>",
                    "<sealed fluid varying object>",
                    "<open flow controlling object>",
                    "<space access object>",
                    "<solid substance flow varying object>",
                    "<mechanical movement controlling object>",
                    "<multiple measure controlling object>",
                    "<restricting object>",
                    "<electricity restricting object>",
                    "<electricity stabilising object>",
                    "<signal stabilising object>",
                    "<movement restricting object>",
                    "<return flow restricting object>",
                    "<flow restrictor>",
                    "<local climate stabilising object>",
                    "<access restricting object>",
                    "<human interaction object>",
                    "<face interaction object>",
                    "<hand interaction object>",
                    "<foot interaction object>",
                    "<finger interaction object>",
                    "<movement interaction object>",
                    "<multi interaction object>",
                    "<transforming object>",
                    "<electric energy transforming object>",
                    "<electric energy converting object>",
                    "<universal power supply>",
                    "<signal converting object>",
                    "<mechanical energy transforming object>",
                    "<mass reduction object>",
                    "<matter reshaping object>",
                    "<organic plant>",
                    "<holding object>",
                    "<positioning object>",
                    "<carrying object>",
                    "<enclosing object>",
                    "<structural supporting object>",
                    "<reinforcing object>",
                    "<framing object>",
                    "<jointing object>",
                    "<fastening object>",
                    "<levelling object>",
                    "<existing ground>",
                    "<guiding object>",
                    "<electric energy guiding object>",
                    "<referencing potential guiding object>",
                    "<electric signal guiding object>",
                    "<light guiding object>",
                    "<sound guiding object>",
                    "<solid matter guiding object>",
                    "<open enclosure guiding object>",
                    "<closed enclosure guiding object>",
                    "<mechanical energy guiding object>",
                    "<rail object>",
                    "<thermal energy guiding object>",
                    "<multiple flow guiding object>",
                    "<high voltage electric energy guiding object>",
                    "<low voltage electric energy guiding object>",
                    "<interfacing object>",
                    "<high voltage connecting object>",
                    "<low voltage connecting object>",
                    "<potential connecting object>",
                    "<electric signal connecting object>",
                    "<light collecting object>",
                    "<collecting interfacing object>",
                    "<sealed flow connecting object>",
                    "<non detachable coupling>",
                    "<detachable coupling>",
                    "<level connecting object>",
                    "<space linking object>",
                    "<multiple flow connector object>",
                    "<activity>",
                    "<maintenance activity>",
                    "<adjust>",
                    "<calibrate>",
                    "<diagnose>",
                    "<inspect>",
                    "<replace>",
                    "<repair>",
                    "<service>",
                    "<supporting activity>",
                    "<admin>",
                    "<assemble>",
                    "<isolate>",
                    "<measure>",
                    "<modify>",
                    "<move>",
                    "<operate>",
                    "<perform>",
                    "<teamwork>",
                    "<state>",
                    "<desirable state>",
                    "<normal state>",
                    "<undesirable state>",
                    "<degraded state>",
                    "<failed state>",
                    "<process>",
                    "<desirable process>",
                    "<undesirable process>",
                    "<property>",
                    "<desirable property>",
                    "<undesirable property>",
                ],
                special_tokens=True,
            )
        elif "_2" in conf.dataset_name.split("/")[-1]:
            tokenizer.add_tokens(
                [
                    "<covering object>",
                    "<substance>",
                    "<guiding object>",
                    "<desirable state>",
                    "<generating object>",
                    "<transforming object>",
                    "<matter processing object>",
                    "<process>",
                    "<undesirable property>",
                    "<state>",
                    "<undesirable process>",
                    "<interfacing object>",
                    "<storing object>",
                    "<emitting object>",
                    "<presenting object>",
                    "<maintenance activity>",
                    "<restricting object>",
                    "<supporting activity>",
                    "<desirable property>",
                    "<controlling object>",
                    "<property>",
                    "<human interaction object>",
                    "<activity>",
                    "<driving object>",
                    "<undesirable state>",
                    "<information processing object>",
                    "<physical object>",
                    "<organism>",
                    "<holding object>",
                    "<desirable process>",
                    "<sensing object>",
                    "<protecting object>",
                ],
                special_tokens=True,
            )
        elif "_1" in conf.dataset_name.split("/")[-1]:
            tokenizer.add_tokens(
                [
                    "<physical object>",
                    "<process>",
                    "<property>",
                    "<activity>",
                    "<state>",
                ],
                special_tokens=True,
            )
        elif "_0" in conf.dataset_name.split("/")[-1]:
            # Untyped MaintIE configuration.
            pass
        else:
            raise NotImplementedError("Level not supported yet!")
    else:
        raise NotImplementedError("Dataset not implemented yet!")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_name_or_path,
        config=config,
    )
    # if not conf.finetune:
    model.resize_token_embeddings(len(tokenizer))

    # data module declaration
    pl_data_module = BasePLDataModule(conf, tokenizer, model)

    # main module declaration
    pl_module = BasePLModule(conf, config, tokenizer, model)

    pl_module = BasePLModule.load_from_checkpoint(
        checkpoint_path=conf.checkpoint_path,
        config=config,
        tokenizer=tokenizer,
        model=model,
    )

    # pl_module.hparams.predict_with_generate = True
    pl_module.hparams.test_file = pl_data_module.conf.test_file
    # trainer
    trainer = pl.Trainer(
        # gpus=conf.gpus,
    )
    # Manually run prep methods on DataModule
    pl_data_module.prepare_data()
    pl_data_module.setup(stage="test")

    trainer.test(pl_module, dataloaders=pl_data_module.test_dataloader())


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
