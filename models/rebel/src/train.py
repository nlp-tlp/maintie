import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pl_data_modules import BasePLDataModule
from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.wandb import WandbLogger

from pytorch_lightning.callbacks import LearningRateMonitor
from generate_samples import GenerateTextSamplesCallback

maintie_general_entities = [
    "<num>",
    "<id>",
    "<date>",
    "<sensitive>",
]

maintie_level_1_unique_entities = [
    "<physical object>",
    "<process>",
    "<property>",
    "<activity>",
    "<state>",
]
maintie_level_2_unique_entities = [
    "<covering object>",
    "<substance>",
    "<guiding object>",
    "<desirable state>",
    "<generating object>",
    "<transforming object>",
    "<matter processing object>",
    "<undesirable property>",
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
    "<human interaction object>",
    "<driving object>",
    "<undesirable state>",
    "<information processing object>",
    "<organism>",
    "<holding object>",
    "<desirable process>",
    "<sensing object>",
    "<protecting object>",
]
maintie_level_3_unique_entities = [
    "<gas>",
    "<liquid>",
    "<solid>",
    "<mixture>",
    "<person>",
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
    "<capacitive storing object>",
    "<inductive storing object>",
    "<electrochemical storing object>",
    "<information storing object>",
    "<open stationary storing object>",
    "<enclosed stationary storing object>",
    "<moveable storing object>",
    "<thermal energy storing object>",
    "<mechanical energy storing object>",
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
    "<mechanical to electrical energy generating object>",
    "<chemical to electrical energy generating object>",
    "<solar to electrical energy generating object>",
    "<signal generating object>",
    "<continuous transfer object>",
    "<discontinuous transfer object>",
    "<liquid flow generating object>",
    "<gaseous flow generating object>",
    "<solar to thermal energy generating object>",
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
    "<electric signal processing object>",
    "<electric signal relaying object>",
    "<optical signalling object>",
    "<fluid signalling object>",
    "<mechanical signalling object>",
    "<multiple kind signalling object>",
    "<electromagnetic rotational driving object>",
    "<electromagnetic linear driving object>",
    "<magnetic force driving object>",
    "<piezoelectric driving object>",
    "<mechanical energy driving object>",
    "<fluid powered driving object>",
    "<combustion engine>",
    "<heat engine>",
    "<infilling object>",
    "<closure object>",
    "<finishing object>",
    "<terminating object>",
    "<hiding object>",
    "<visible state indicator>",
    "<scalar display>",
    "<graphical display>",
    "<acoustic device>",
    "<tactile device>",
    "<ornamental object>",
    "<multiple form presenting object>",
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
    "<electricity restricting object>",
    "<electricity stabilising object>",
    "<signal stabilising object>",
    "<movement restricting object>",
    "<return flow restricting object>",
    "<flow restrictor>",
    "<local climate stabilising object>",
    "<access restricting object>",
    "<face interaction object>",
    "<hand interaction object>",
    "<foot interaction object>",
    "<finger interaction object>",
    "<movement interaction object>",
    "<multi interaction object>",
    "<electric energy transforming object>",
    "<electric energy converting object>",
    "<universal power supply>",
    "<signal converting object>",
    "<mechanical energy transforming object>",
    "<mass reduction object>",
    "<matter reshaping object>",
    "<organic plant>",
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
    "<adjust>",
    "<calibrate>",
    "<diagnose>",
    "<inspect>",
    "<replace>",
    "<repair>",
    "<service>",
    "<admin>",
    "<assemble>",
    "<isolate>",
    "<measure>",
    "<modify>",
    "<move>",
    "<operate>",
    "<perform>",
    "<teamwork>",
    "<normal state>",
    "<degraded state>",
    "<failed state>",
]


def train(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)

    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id=0,
        early_stopping=False,
        no_repeat_ngram_size=0,
        dropout=conf.dropout,
        forced_bos_token_id=None,
    )

    using_maintie_base_model = "maintie" in conf.model_name_or_path.split("/")[-1]
    print(f"conf.model_name_or_path: {conf.model_name_or_path.split('/')[-1]}")
    print(
        f'{"Using pretrained MaintIE base model" if using_maintie_base_model else "Using REBEL base model"}'
    )

    if using_maintie_base_model:
        # Using custom maintieistant model
        tokenizer_kwargs = {
            "use_fast": conf.use_fast_tokenizer,
            "additional_special_tokens": [
                "<obj>",
                "<subj>",
                "<triplet>",
                *maintie_general_entities,
                *maintie_level_1_unique_entities,
            ],
            # Here the tokens for head and tail are legacy and only needed if finetuning over the public REBEL checkpoint, but are not used. If training from scratch, remove this line and uncomment the next one.
        }
    else:
        tokenizer_kwargs = {
            "use_fast": conf.use_fast_tokenizer,
            "additional_special_tokens": [
                "<obj>",
                "<subj>",
                "<triplet>",
            ],  # Here the tokens for head and tail are legacy and only needed if finetuning over the public REBEL checkpoint, but are not used. If training from scratch, remove this line and uncomment the next one.
            #         "additional_special_tokens": ['<obj>', '<subj>', '<triplet>'],
        }

    tokenizer = AutoTokenizer.from_pretrained(
        conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
        **tokenizer_kwargs,
    )
    print(f"Base tokenizer size: {len(tokenizer)}")

    if conf.dataset_name.split("/")[-1] == "conll04_typed.py":
        tokenizer.add_tokens(
            ["<peop>", "<org>", "<other>", "<loc>"], special_tokens=True
        )
    if "maintie" in conf.dataset_name.split("/")[-1]:
        print("ADDING SPECIAL TOKENS FOR MAINTIE")

        if not using_maintie_base_model:
            print(
                "Not using maintie model, adding <num>, <id>, <date>, <sensitive> tokens."
            )
            tokenizer.add_tokens(
                ["<num>", "<id>", "<date>", "<sensitive>"], special_tokens=True
            )

        if "_3" in conf.dataset_name.split("/")[-1]:
            tokenizer.add_tokens(
                [
                    *(
                        []
                        if using_maintie_base_model
                        else maintie_level_1_unique_entities
                    ),
                    *maintie_level_2_unique_entities,
                    *maintie_level_3_unique_entities,
                ],
                special_tokens=True,
            )
        elif "_2" in conf.dataset_name.split("/")[-1]:
            tokenizer.add_tokens(
                [
                    *(
                        []
                        if using_maintie_base_model
                        else maintie_level_1_unique_entities
                    ),
                    *maintie_level_2_unique_entities,
                ],
                special_tokens=True,
            )

        elif "_1" in conf.dataset_name.split("/")[-1]:
            tokenizer.add_tokens(
                [] if using_maintie_base_model else maintie_level_1_unique_entities,
                special_tokens=True,
            )
        elif "_0" in conf.dataset_name.split("/")[-1]:
            # Untyped MaintIE configuration.
            pass
        else:
            raise NotImplementedError("Level not supported yet!")

    print(f"Final tokenizer size: {len(tokenizer)}")

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

    # wandb_logger = WandbLogger(
    #     project=conf.dataset_name.split("/")[-1].replace(".py", ""),
    #     name=conf.model_name_or_path.split("/")[-1],
    # )

    callbacks_store = []

    if conf.apply_early_stopping:
        callbacks_store.append(
            EarlyStopping(
                monitor=conf.monitor_var,
                mode=conf.monitor_var_mode,
                patience=conf.patience,
            )
        )

    callbacks_store.append(
        ModelCheckpoint(
            monitor=conf.monitor_var,
            # monitor=None,
            dirpath=f"experiments/{conf.model_name}",
            save_top_k=conf.save_top_k,
            verbose=True,
            save_last=True,
            mode=conf.monitor_var_mode,
        )
    )
    # callbacks_store.append(GenerateTextSamplesCallback(conf.samples_interval))    # ERROR: "TypeError: on_train_batch_end() missing 1 required positional argument: 'dataloader_idx'"
    callbacks_store.append(LearningRateMonitor(logging_interval="step"))

    # trainer
    trainer = pl.Trainer(
        # gpus=conf.gpus,
        accumulate_grad_batches=conf.gradient_acc_steps,
        gradient_clip_val=conf.gradient_clip_value,
        val_check_interval=conf.val_check_interval,
        callbacks=callbacks_store,
        max_steps=conf.max_steps,
        # max_steps=total_steps,
        precision=conf.precision,
        # amp_level=conf.amp_level,
        # logger=wandb_logger,
        # resume_from_checkpoint=conf.checkpoint_path,
        limit_val_batches=conf.val_percent_check,
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module, ckpt_path=conf.checkpoint_path)


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
