import argparse
import numpy as np
import os
import random
import torch
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from processing.sampler import ConcatDatasetBatchSampler
from processing.datasets import HDF5_dataset, ConcatDatasetUrban
from nnet.CRNN import CRNN
from utils.encoder import ManyHotEncoder
from utils.schedulers import ExponentialWarmup
from MT_trainer import CoSMo_benchmark
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
    noisy=False,
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
    encoder = ManyHotEncoder(
        taxonomy,
        use_taxo_fine=config["training"]["train_on_fine_taxo"],
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["features"]["n_filters"],
        frame_hop=config["features"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    config.update({"log_dir": log_dir})
    config["net"].update({"n_class": len(encoder.labels)})
    config.update({"taxonomy": encoder.taxonomy["name"]})
    config.update({"noisy": noisy})
    config["training"].update({"scheduler_type": "none"})

    if config["features"]["transform_type"] == "log+pcen":
        config["net"].update({"n_in_channel": 2})

    ##### model definition  ############
    sed_student = CRNN(**config["net"])

    if test_state_dict is None:
        ##### data prep train valid ##########
        SINGAPURA_train_set = HDF5_dataset(
            config["data"]["root_path"] + config["data"]["hdf5_train"],
            "SINGA-PURA",
            encoder,
        )
        unlabelled_SINGAPURA_train_set = HDF5_dataset(
            config["data"]["root_path"] + config["data"]["hdf5_train"],
            "unlabelled_SINGA-PURA",
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
        if batch_sizes[2] > 0:
            tot_train_data.append(unlabelled_SINGAPURA_train_set)
            bs.append(batch_sizes[2])

        train_dataset = ConcatDatasetUrban(
            tot_train_data, encoder, batch_sizes=batch_sizes
        )
        samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
        batch_sampler = ConcatDatasetBatchSampler(samplers, bs)

        valid_dataset = ConcatDatasetUrban(
            [SINGAPURA_val_set, SONYC_val_set],
            encoder,
            [config["training"]["batch_size_val"]],
        )

        test_dataset = ConcatDatasetUrban(
            [SINGAPURA_test_set, SONYC_test_set],
            encoder,
            [config["training"]["batch_size_val"]],
            analyse_proximity=True,
        )

        ##### training params and optimizers ############
        epoch_len = min(
            [
                len(tot_train_data[indx])
                // (
                    config["training"]["batch_size"][indx]
                    * config["training"]["accumulate_batches"]
                    + 1e-15
                )
                for indx in range(len(tot_train_data))
            ]
        )

        opt = torch.optim.Adam(sed_student.parameters(), lr=config["opt"]["lr"])

        # If the train split is too small, exp_steps must be clipped
        # exp_steps = config["training"]["n_steps_warmup"]
        exp_steps = int(
            config["training"]["n_epochs_warmup"]
            * config["training"].get("limit_train_batches")
            * 32
            / np.max(config["training"]["batch_size"][:2])
        )
        config["training"].update({"n_steps_warmup": exp_steps})

        if config["training"]["scheduler_type"] == "none":
            exp_scheduler = None
        elif config["training"]["scheduler_type"] == "rampup":
            exp_scheduler = ExponentialWarmup(
                opt, max_lr=config["opt"]["lr"], rampup_length=exp_steps
            )
        elif config["training"]["scheduler_type"] == "cycle":
            exp_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=config["opt"]["lr"], total_steps=exp_steps * 3
            )
        logger = TensorBoardLogger(
            os.path.dirname(config["log_dir"]), name=config["experiment_name"]
        )
        logger.log_hyperparams(config)
        print(f"experiment dir: {logger.log_dir}")

        """
        callbacks = [
            EarlyStopping(
                monitor="val/obj_metric",
                patience=config["training"]["early_stop_patience"],
                verbose=True,
                mode="max",
            ),
            ModelCheckpoint(
                logger.log_dir,
                monitor="val/obj_metric",
                save_top_k=1,
                mode="max",
                save_last=True,
            ),
            StochasticWeightAveraging(
                swa_lrs=1e-2
            ),
        ],
        """
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

        precision = 32

    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None
        opt = None
        exp_scheduler = None
        logger = True
        callbacks = None
        SINGAPURA_test_set = HDF5_dataset(
            config["data"]["root_path"] + config["data"]["hdf5_test"],
            "SINGA-PURA",
            encoder,
        )
        SONYC_test_set = HDF5_dataset(
            config["data"]["root_path"] + config["data"]["hdf5_test"], "SONYC", encoder
        )
        test_dataset = ConcatDatasetUrban(
            [SINGAPURA_test_set, SONYC_test_set],
            encoder,
            [config["training"]["batch_size_val"]],
            analyse_proximity=True,
        )

    urban_detection_training = CoSMo_benchmark(
        config,
        encoder=encoder,
        sed_student=sed_student,
        opt=opt,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        scheduler=exp_scheduler,
        fast_dev_run=fast_dev_run,
        evaluation=evaluation,
        noisy=noisy,
        precision=precision,
    )

    # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
    if fast_dev_run:
        log_every_n_steps = 1
        limit_train_batches = 30
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = 3
    else:
        log_every_n_steps = 40
        limit_train_batches = int(
            config["training"].get("limit_train_batches")
            * 32
            / np.max(config["training"]["batch_size"][:2])
        )
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = config["training"]["n_epochs"]

    print(np.max(config["training"]["batch_size"][:2]))
    print(f"using {limit_train_batches} steps per epoch")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=n_epochs,
        callbacks=callbacks,
        precision=precision,
        accelerator="gpu",
        devices=1,
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
    )

    if test_state_dict is None:
        trainer.fit(urban_detection_training)
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best model: {best_path}")
        test_state_dict = torch.load(best_path)["state_dict"]

    urban_detection_training.load_state_dict(test_state_dict)
    trainer.test(urban_detection_training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training a SED system for Urban Sounds Datasets")
    parser.add_argument(
        "--conf_file",
        default="../config/sed.yaml",
        help="The configuration file with all the experiment parameters.",
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
        "--noisy",
        action="store_true",
        default=False,
        help="add noise to the input features of the teacher",
    )
    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        config = yaml.safe_load(f)

    # based on config file, we open the taxonomy file
    with open(config["data"]["taxonomy_path"], "r") as f:
        taxonomy = yaml.safe_load(f)

    evaluation = False
    test_from_checkpoint = args.test_from_checkpoint

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

    seed = config["training"]["seed"] * 14
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
        args.noisy,
    )
