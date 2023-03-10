import argparse
import numpy as np
import os
import random
import torch
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import madgrad

from processing.sampler import ConcatDatasetBatchSampler
from processing.datasets_hear import HDF5_dataset, ConcatDatasetUrban
from nnet.hear_models import HearDetector
from utils.encoder import ManyHotEncoder
from classif_trainer import HearTrainer
from loguru import logger as lg


@lg.catch
def single_run(
    config,
    taxonomy,
    log_dir,
    checkpoint_resume=None,
    test_state_dict=None,
    fast_dev_run=False,
    evaluation=False,
):
    """
    Running sound event detection baseline

    Args:
        config (dict): the dictionary of configuration params
        log_dir (str): path to log directory
        gpus (int): number of gpus to use
        checkpoint_resume (str, optional): path to checkpoint to resume from. Defaults to "".
        test_state_dict (dict, optional): if not None, no training is involved. This dictionary is the state_dict
            to be loaded to test the model.
        fast_dev_run (bool, optional): whether to use a run with only one batch at train and validation, useful
            for development purposes.
    """

    ##### data prep test ##########
    if "passt" in config["net"]["hear_encoder"]:
        hop = 398
    elif "open_l3" in config["net"]["hear_encoder"]:
        hop = 512
    else:
        raise NotImplementedError(
            f"encoder {config['net']['hear_encoder']} is not supported"
        )
    encoder = ManyHotEncoder(
        taxonomy,
        use_taxo_fine=config["training"]["train_on_fine_taxo"],
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["features"]["n_filters"],
        frame_hop=hop,
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    config.update({"log_dir": log_dir})
    config["net"].update({"n_class": len(encoder.labels)})
    config.update({"taxonomy": encoder.taxonomy["name"]})

    ##### model definition  ############
    sed_hear = HearDetector(**config["net"])
    if test_state_dict is None:
        ##### data prep train valid ##########
        SINGAPURA_train_set = HDF5_dataset(
            config["data"]["root_path"] + config["data"]["hdf5_train"],
            "SINGA-PURA",
            encoder,
        )
        SONYC_train_set = HDF5_dataset(
            config["data"]["root_path"] + config["data"]["hdf5_train"], "SONYC", encoder
        )
        SINGAPURA_val_set = HDF5_dataset(
            config["data"]["root_path"] + config["data"]["hdf5_val"],
            "SINGA-PURA",
            encoder,
        )
        SONYC_val_set = HDF5_dataset(
            config["data"]["root_path"] + config["data"]["hdf5_val"], "SONYC", encoder
        )
        SINGAPURA_test_set = HDF5_dataset(
            config["data"]["root_path"] + config["data"]["hdf5_test"],
            "SINGA-PURA",
            encoder,
        )
        SONYC_test_set = HDF5_dataset(
            config["data"]["root_path"] + config["data"]["hdf5_test"], "SONYC", encoder
        )

        batch_sizes = config["training"]["batch_size"]
        bs = []
        tot_train_data = []
        if batch_sizes[0] > 0:
            tot_train_data.append(SINGAPURA_train_set)
            bs.append(batch_sizes[0])
        if batch_sizes[1] > 0:
            tot_train_data.append(SONYC_train_set)
            bs.append(batch_sizes[1])

        train_dataset = ConcatDatasetUrban(
            tot_train_data,
            encoder,
            batch_sizes=batch_sizes,
            hear_encoder=config["net"]["hear_encoder"],
        )
        samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
        batch_sampler = ConcatDatasetBatchSampler(samplers, bs)

        valid_dataset = ConcatDatasetUrban(
            [SINGAPURA_val_set, SONYC_val_set],
            encoder,
            [config["training"]["batch_size_val"]],
            hear_encoder=config["net"]["hear_encoder"],
        )

        test_dataset = ConcatDatasetUrban(
            [SINGAPURA_test_set, SONYC_test_set],
            encoder,
            [config["training"]["batch_size_val"]],
            hear_encoder=config["net"]["hear_encoder"],
            analyse_proximity=True,
        )

        ##### training params and optimizers ############
        if config["opt"]["name"] == "MADGRAD":
            opt = madgrad.MADGRAD(sed_hear.parameters(), lr=config["opt"]["lr"])
        else:
            opt = torch.optim.Adam(sed_hear.parameters(), lr=config["opt"]["lr"])

        print(os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1])
        logger = TensorBoardLogger(
            os.path.dirname(config["log_dir"]), name=config["experiment_name"]
        )
        logger.log_hyperparams(config)
        print(f"experiment dir: {logger.log_dir}")

        callbacks = [
            EarlyStopping(
                monitor="Val/obj_metric",
                patience=config["training"]["early_stop_patience"],
                verbose=True,
                mode="max",
            ),
            ModelCheckpoint(
                logger.log_dir,
                monitor="Val/obj_metric",
                save_top_k=1,
                mode="max",
                save_last=True,
            ),
        ]
    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None
        opt = None
        logger = True
        callbacks = None

    print("init trainer")
    urban_detection_training = HearTrainer(
        config,
        encoder=encoder,
        model=sed_hear,
        opt=opt,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        fast_dev_run=fast_dev_run,
        evaluation=evaluation,
    )

    # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
    if fast_dev_run:
        flush_logs_every_n_steps = 1
        log_every_n_steps = 1
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 2
        n_epochs = 3
    else:
        flush_logs_every_n_steps = 100
        log_every_n_steps = 40
        limit_train_batches = int(
            config["training"].get("limit_train_batches")
            * 32
            / np.max(config["training"]["batch_size"][:2])
        )
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = config["training"]["n_epochs"]

    torch.backends.cudnn.benchmark = True
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=callbacks,
        gpus=1,
        accelerator="gpu",
        strategy=config["training"].get("backend"),
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        resume_from_checkpoint=checkpoint_resume,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        deterministic=True,
    )

    if test_state_dict is None:
        trainer.fit(urban_detection_training)
        print("training finished")
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best model: {best_path}")
        test_state_dict = torch.load(best_path)["state_dict"]

    urban_detection_training.load_state_dict(test_state_dict)
    trainer.test(urban_detection_training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training a SED system for Urban Sounds Datasets")
    parser.add_argument(
        "--conf_file",
        default="../config/sed_hear.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--taxonomy",
        default="../config/taxonomy_SONYC.yaml",
        help="The taxonomy to use along with the mapping between each dataset labels and the unified taxonomy.",
    )
    parser.add_argument(
        "--log_dir",
        default="../experiments/COSMO/",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).",
    )
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified"
    )

    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
        "It uses very few batches and epochs so it won't give any meaningful result.",
    )

    parser.add_argument(
        "--eval_from_checkpoint", default=None, help="Evaluate the model specified"
    )

    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        config = yaml.safe_load(f)

    with open(args.taxonomy, "r") as f:
        taxonomy = yaml.safe_load(f)

    evaluation = False
    test_from_checkpoint = args.test_from_checkpoint

    if args.eval_from_checkpoint is not None:
        test_from_checkpoint = args.eval_from_checkpoint
        evaluation = True

    test_model_state_dict = None
    if test_from_checkpoint is not None:
        checkpoint = torch.load(test_from_checkpoint)
        config_ckpt = checkpoint["hyper_parameters"]
        config_ckpt["data"] = config["data"]
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]

    if evaluation:
        config["training"]["batch_size_val"] = 1

    seed = config["training"]["seed"] * 26
    config["training"].update({"seed": seed})
    if seed:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)
    test_only = test_from_checkpoint is not None
    single_run(
        config,
        taxonomy,
        args.log_dir,
        args.resume_from_checkpoint,
        test_model_state_dict,
        args.fast_dev_run,
        evaluation,
    )
