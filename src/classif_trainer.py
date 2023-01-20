import os
import random
import warnings
import random
from copy import deepcopy
import torchmetrics
import sklearn.metrics as M
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger as lg
from utils.utils import(
    batched_decode_preds, 
    log_sedeval_metrics,
    focal_loss,
)
from utils.mme_eval import multi_eval
from utils.evaluation_measures import(
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
)

@lg.catch
class HearTrainer(pl.LightningModule):
    """ Pytorch lightning module for the SED 2021 baseline
    Args:
        hparams: dict, the dictionary to be used for the current experiment/
        encoder: ManyHotEncoder object, object to encode and decode labels.
        model: torch.Module, the model (hear embeddings + classifier) to be trained. Only the classifier head is trained.
        opt: torch.optimizer.Optimizer object, the optimizer to be used
        train_data: torch.utils.data.Dataset subclass object, the training data to be used.
        valid_data: torch.utils.data.Dataset subclass object, the validation data to be used.
        test_data: torch.utils.data.Dataset subclass object, the test data to be used.
        train_sampler: torch.utils.data.Sampler subclass object, the sampler to be used in the training dataloader.
        fast_dev_run: bool, whether to launch a run with only one batch for each set, this is for development purpose,
            to test the code runs.
    """

    def __init__(
        self,
        hparams,
        encoder,
        model,
        opt=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        train_sampler=None,
        fast_dev_run=False,
        evaluation=False,
    ):
        super().__init__()
        self.hparams.update(hparams)

        self.encoder = encoder
        self.model = model
        self.opt = opt
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_sampler = train_sampler
        self.fast_dev_run = fast_dev_run
        self.evaluation=evaluation
        if self.fast_dev_run:
            self.num_workers = 1
        else:
            self.num_workers = self.hparams["training"]["num_workers"]


        # instantiating losses
        if self.hparams["training"]["loss"] == "focal":
            self.supervised_loss = focal_loss
        else:
            self.supervised_loss = torch.nn.BCELoss()
        self.weight_loss_sup = hparams["training"]["weight_loss_sup"]

        self._init_metrics()


    def forward(self, embed):
        return self.model(embed)
    
    
    def on_train_start(self):
        warnings.filterwarnings(
            "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
        )
        warnings.filterwarnings(
            "ignore", ".*invalid value encountered in divide*"
        )
        warnings.filterwarnings(
            "ignore", ".*Average precision score for one or more classes*"
        )
        warnings.filterwarnings(
            "ignore", ".*No positive samples in targets,*"
        )
        
        warnings.filterwarnings(
            "ignore", ".*which may be a bottleneck.*"
        )
        
        warnings.filterwarnings(
            "ignore", ".*To copy construct from a tensor*"
        )
        
        warnings.filterwarnings(
            "ignore", ".*A similar operating point exists*"
        )
    
    
    def training_step(self, batch, batch_indx):
        """ Apply the training for one batch (a step). Used during trainer.fit

        Args:
            batch: torch.Tensor, batch input tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.

        Returns:
           torch.Tensor, the loss to take into account.
        """

        audio, labels = batch
        indx_strong, indx_weak, indx_unlabelled = self.hparams["training"]["batch_size"]

        batch_num = audio.shape[0]
        
        # deriving masks for each dataset
        strong_mask = torch.zeros(batch_num, device=self.device).bool()
        weak_mask = torch.zeros(batch_num, device=self.device).bool()
        unlabelled_mask = torch.zeros(batch_num, device=self.device).bool()
        strong_mask[:indx_strong] = 1
        weak_mask[indx_strong : indx_weak + indx_strong] = 1
        unlabelled_mask[-indx_unlabelled:] = 1
        # deriving strong and weak labels
        labels_weak = torch.max(labels, -1)[0].float()
        labels_weak_SONYC = labels_weak[weak_mask]
        labels_weak_SGP = labels_weak[strong_mask]
        labels_strong_SGP = labels[strong_mask]
        
        # sed forward
        strong_preds, weak_preds = self.model(audio)            

        # supervised loss on strong labels
        loss_strong = self.supervised_loss(
            strong_preds[strong_mask], labels_strong_SGP
        )
        # supervised loss on weakly labelled
        loss_weak = self.supervised_loss(weak_preds[weak_mask], labels_weak_SONYC)
        
        # supervised loss on strongly labelled turned into weakly
        loss_weakstrong = self.supervised_loss(weak_preds[strong_mask], labels_weak_SGP)
        
        
        # total supervised loss
        tot_loss = (loss_strong.nan_to_num() * self.weight_loss_sup[0] + 
                              loss_weak.nan_to_num() * self.weight_loss_sup[1]+ 
                              loss_weakstrong.nan_to_num() * self.weight_loss_sup[2])         
        
        self.log("Train/loss_strong", loss_strong)
        self.log("Train/loss_weak", loss_weak)
        self.log("Train/loss_weakstrong", loss_weakstrong)
        self.log("Train/step", self.global_step, prog_bar=True)
        self.log("Train/lr", self.opt.param_groups[-1]["lr"], prog_bar=True)

        return tot_loss

    def validation_step(self, batch, batch_indx):
        """ Apply validation to a batch (step). Used during trainer.fit

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, filenames = batch

        strong_preds, weak_preds = self.model(audio)

        # we derive masks for each dataset based on filenames
        mask_SONYC = (
            torch.tensor(
                [
                    x[0] != '['
                    for x in filenames
                ]
            ,device=self.device)
            .detach()
            .bool()
        )
        mask_SGP = (
            torch.tensor(
                [
                    x[0] == '['
                    for x in filenames
                ]
            ,device=self.device)
            .detach()
            .bool()
        )        
        
        if torch.any(mask_SONYC):
            
            labels_weak_SONYC = (torch.sum(labels[mask_SONYC], -1) >= 1).float()
            loss_weak = self.supervised_loss(
                weak_preds[mask_SONYC], labels_weak_SONYC 
            )

            self.log("Val/loss_weak_SONYC", loss_weak.detach())

        if torch.any(mask_SGP):
            labels_weak_SGP = (torch.sum(labels[mask_SGP], -1) > 0).float()
            # supervised loss on strongly labelled turned into weakly
            loss_strong = self.supervised_loss(
                strong_preds[mask_SGP], labels[mask_SGP]
            )

            loss_weakstrong = self.supervised_loss(
                weak_preds[mask_SGP], labels_weak_SGP
            )

            self.log("Val/loss_strong_SGP", loss_strong.detach())
            self.log("Val/loss_weak_SGP", loss_weakstrong.detach())

        # Convert predictions and groundtruths to the coarse taxonomy
        if self.hparams["training"]["train_on_fine_taxo"]:
            strong_preds = self.encoder.fine_to_coarse(strong_preds)
            weak_preds = self.encoder.fine_to_coarse(weak_preds)
            labels = self.encoder.fine_to_coarse(labels)
        
        self.update_metrics("Val", strong_preds, weak_preds, labels, filenames, mask_SGP, mask_SONYC)
        
        return
    
    def validation_epoch_end(self, outputs):
        """ Function applied at the end of all the validation steps of the epoch.

        Args:
            outputs: torch.Tensor, the concatenation of everything returned by validation_step.

        Returns:
            torch.Tensor, the objective metric to be used to choose the best model from for example.
        """

        self.compute_metrics("Val",
                            self.valid_data.datasets,
                            self.hparams["training"]["batch_size_val"],
                            self.hparams["training"]["batch_size_val"])
        self.reset_metrics("Val")   
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model"] = self.model.state_dict()
        return checkpoint

    def test_step(self, batch, batch_indx):
        """ Apply Test to a batch (step), used only when (trainer.test is called)

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """
        
        audio, labels, filenames = batch        

        # Because we use proximity one-hot encoding we need to convert them to one-hot encoding depending on the proximity
        labels_near = torch.logical_or(labels == 2, labels == 5).float()
        labels_far = (labels == 3).float()
        labels_all = labels.bool().float()
        labels_weak_near = torch.max(labels_near, -1)[0].float()
        labels_weak_far = torch.max(labels_far, -1)[0].float()

        # we derive masks for each dataset based on filenames
        mask_SONYC = (
            torch.tensor(
                [
                    x[0] != '['
                    for x in filenames
                ]
            ,device=self.device)
            .bool()
        )
        mask_SGP = (
            torch.tensor(
                [
                    x[0] == '['
                    for x in filenames
                ]
            ,device=self.device)
            .detach()
            .bool()
        )
        
        strong_preds, weak_preds = self.model(audio)
        
        # Apply proximity masking to student predictions (we want to ignore the events (far or near) not considered by the given proximity)
        strong_preds_near = strong_preds * torch.logical_not(labels_far)
        strong_preds_far = strong_preds * torch.logical_not(labels_near)
        labels_far = labels_far * torch.logical_not(labels_near)
        labels_near = labels_near * torch.logical_not(labels_far)
        
        weak_preds_near = weak_preds * torch.logical_not(labels_weak_far)
        weak_preds_far = weak_preds * torch.logical_not(labels_weak_near)
        labels_weak_far = labels_weak_far * torch.logical_not(labels_weak_near)
        labels_weak_near = labels_weak_near * torch.logical_not(labels_weak_far)
        
        # we need to compute masks by proximity
        mask_near = torch.any(labels_weak_near, axis=1)
        mask_far = torch.any(labels_weak_far, axis=1)
        mask_near_SONYC = torch.logical_and(mask_near, mask_SONYC)
        mask_far_SONYC = torch.logical_and(mask_far, mask_SONYC)
        mask_near_SGP = torch.logical_and(mask_near, mask_SGP)
        mask_far_SGP = torch.logical_and(mask_far, mask_SGP)

        if torch.any(mask_SGP):
            labels_weak_SGP = (torch.sum(labels[mask_SGP], -1) > 0).float()
            loss_strong = self.supervised_loss(strong_preds[mask_SGP], labels_all[mask_SGP])

            loss_weakstrong = self.supervised_loss(
                weak_preds[mask_SGP], labels_weak_SGP
            )            
            
            self.log("Test/loss_strong_SGP", loss_strong.detach())
            self.log("Test/loss_weak_SGP", loss_weakstrong.detach())
        
        if torch.any(mask_SONYC):
            labels_weak_SONYC = labels_all[mask_SONYC, : , 0]
            loss_weak = self.supervised_loss(
                weak_preds[mask_SONYC], labels_weak_SONYC 
            )

            self.log("Test/loss_weak_SONYC", loss_weak.detach())
        

        # Convert predictions and groundtruths to the coarse taxonomy
        if self.hparams["training"]["train_on_fine_taxo"]:
            strong_preds = self.encoder.fine_to_coarse(strong_preds)
            strong_preds_near = self.encoder.fine_to_coarse(strong_preds_near)
            strong_preds_far = self.encoder.fine_to_coarse(strong_preds_far)
            weak_preds = self.encoder.fine_to_coarse(weak_preds)
            weak_preds_near = self.encoder.fine_to_coarse(weak_preds_near)
            weak_preds_far = self.encoder.fine_to_coarse(weak_preds_far)
            labels_near = self.encoder.fine_to_coarse(labels_near)
            labels_far = self.encoder.fine_to_coarse(labels_far)
            labels_all = self.encoder.fine_to_coarse(labels_all)
            labels_weak_near = self.encoder.fine_to_coarse(labels_weak_near)
            labels_weak_far = self.encoder.fine_to_coarse(labels_weak_far)        

        # update metrics by proximity
        if torch.any(mask_near_SONYC):
            self.weak_metrics["Test"]["SONYC_near"].update(weak_preds_near[mask_near_SONYC], labels_weak_near[mask_near_SONYC].int())
            self.weak_metrics_by_class["SONYC_near"].update(weak_preds_near[mask_near_SONYC], labels_weak_near[mask_near_SONYC].int())
            self.update_df_metrics(weak_preds_near[mask_near_SONYC], labels_weak_near[mask_near_SONYC].int(), "SONYC", "near")
        if torch.any(mask_far_SONYC):
            self.weak_metrics["Test"]["SONYC_far"].update(weak_preds_far[mask_far_SONYC], labels_weak_far[mask_far_SONYC].int())
            self.weak_metrics_by_class["SONYC_far"].update(weak_preds_far[mask_far_SONYC], labels_weak_far[mask_far_SONYC].int())
            self.update_df_metrics(weak_preds_far[mask_far_SONYC], labels_weak_far[mask_far_SONYC].int(), "SONYC", "far")

        if torch.any(mask_near_SGP):
            self.weak_metrics["Test"]["SGP_near"].update(weak_preds_near[mask_near_SGP], labels_weak_near[mask_near_SGP].int())
            self.weak_metrics_by_class["SGP_near"].update(weak_preds_near[mask_near_SGP], labels_weak_near[mask_near_SGP].int())
            self.update_df_metrics(weak_preds_near[mask_near_SGP], labels_weak_near[mask_near_SGP].int(), "SGP", "near")

            filenames_strong_near = filenames[mask_near_SGP.cpu().numpy()]

            decoded_strong_near = batched_decode_preds(
                strong_preds_near[mask_near_SGP],
                filenames_strong_near,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.strong_preds_buffers["Test_near"].keys()),
            )
            
            for th in self.strong_preds_buffers["Test_near"].keys():
                self.strong_preds_buffers["Test_near"][th] = pd.concat(
                    [self.strong_preds_buffers["Test_near"][th], decoded_strong_near[th]]
                    , ignore_index=True
                )

        if torch.any(mask_far_SGP):
            self.weak_metrics["Test"]["SGP_far"].update(weak_preds_far[mask_far_SGP], labels_weak_far[mask_far_SGP].int())
            self.weak_metrics_by_class["SGP_far"].update(weak_preds_far[mask_far_SGP], labels_weak_far[mask_far_SGP].int())
            self.update_df_metrics(weak_preds_far[mask_far_SGP], labels_weak_far[mask_far_SGP].int(), "SGP", "far")

            filenames_strong_far = filenames[mask_far_SGP.cpu().numpy()]

            decoded_strong_far = batched_decode_preds(
                strong_preds_far[mask_far_SGP],
                filenames_strong_far,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.strong_preds_buffers["Test_far"].keys()),
            )
            
            for th in self.strong_preds_buffers["Test_far"].keys():
                self.strong_preds_buffers["Test_far"][th] = pd.concat(
                    [self.strong_preds_buffers["Test_far"][th], decoded_strong_far[th]]
                    , ignore_index=True
                )
            
            
        if torch.any(mask_SGP):

            filenames_strong = [
                x
                for x in filenames
                if x[0] == '['
            ]
            
            
            # compute psds
            decoded_strong = batched_decode_preds(
                strong_preds[mask_SGP],
                filenames_strong,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.strong_preds_buffers["test_psds"].keys()),
            )

            for th in self.strong_preds_buffers["test_psds"].keys():
                self.strong_preds_buffers["test_psds"][th] = pd.concat(
                    [self.strong_preds_buffers["test_psds"][th], decoded_strong[th]]
                    , ignore_index=True
                )

        self.update_metrics("Test", strong_preds, weak_preds, labels_all, filenames, mask_SGP, mask_SONYC)


    def on_test_epoch_end(self):
        try:
            log_dir = self.logger.log_dir
        except Exception as e:
            log_dir = self.hparams["log_dir"]
        save_dir = os.path.join(log_dir, "metrics_test")
        os.mkdir(save_dir)
        
        self.compute_metrics("Test",
                            self.test_data.datasets,
                            self.hparams["training"]["batch_size_val"],
                            self.hparams["training"]["batch_size_val"],
                            save_dir=save_dir    
        )
        

        # calculate the psds metrics
        psds_score_scenario1 = compute_psds_from_operating_points(
            self.strong_preds_buffers["test_psds"],
            self.test_data.datasets[0].groundtruths,
            self.test_data.datasets[0].durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_ct=0,
            alpha_st=0,
            # In our case, some classes are too scarce or difficult to detect so
            # alpha_st is set to 0 to minimize the influence of hard classes on psds score.
            save_dir=os.path.join(save_dir, "scenario1"),
        )

        psds_score_scenario2 = compute_psds_from_operating_points(
            self.strong_preds_buffers["test_psds"],
            self.test_data.datasets[0].groundtruths,
            self.test_data.datasets[0].durations,
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=0,
            # In our case, some classes are too scarce or difficult to detect so
            # alpha_st is set to 0 to minimize the influence of hard classes on psds score.
            save_dir=os.path.join(save_dir, "scenario2"),
        )
        
        best_test_result = torch.tensor(max(psds_score_scenario1, psds_score_scenario2)).detach()

        results = {
            "hp_metric": best_test_result,
            "Test/strong_SGP_psds_score_scenario1": psds_score_scenario1,
            "Test/strong_SGP_psds_score_scenario2": psds_score_scenario2,
        }
        if self.logger is not None:             
            self.logger.log_metrics(results)
            self.logger.log_hyperparams(self.hparams, results)

        for key in results.keys():
            self.log(key, results[key], prog_bar=True, logger=False)


        # compute strong metrics for SGP near, SGP far, SGP_monoph, SGP_lowpolyph, SGP_highpolyph
        
        for subset in ["near", "far", "monoph", "lowpolyph", "highpolyph"]:
            if subset == "near":
                buffer = self.strong_preds_buffers["Test_near"]
                groundtruths = self.test_data.datasets[0].groundtruths_near 
                durations = self.test_data.datasets[0].durations_near
            elif subset == "far":
                buffer = self.strong_preds_buffers["Test_far"]
                groundtruths = self.test_data.datasets[0].groundtruths_far
                durations = self.test_data.datasets[0].durations_far
            elif subset == "monoph":
                buffer = self.strong_preds_buffers["Test_monoph"]
                groundtruths = self.test_data.datasets[0].groundtruths_monoph
                durations = self.test_data.datasets[0].durations_monoph
            elif subset == "lowpolyph":
                buffer = self.strong_preds_buffers["Test_lowpolyph"]
                groundtruths = self.test_data.datasets[0].groundtruths_lowpolyph
                durations = self.test_data.datasets[0].durations_lowpolyph
            elif subset == "highpolyph":
                buffer = self.strong_preds_buffers["Test_highpolyph"]
                groundtruths = self.test_data.datasets[0].groundtruths_highpolyph
                durations = self.test_data.datasets[0].durations_highpolyph
            
            intersection_f1_macro = compute_per_intersection_macro_f1(
                buffer,
                groundtruths,
                durations,
            )

            
            event_f1_macro, event_f1_micro, segment_f1_macro, segment_f1_micro = log_sedeval_metrics(
                buffer[0.5],
                groundtruths,
                save_dir,
            )
            
            '''
            mme_res = multi_eval.get_single_result_df(groundtruths,
                                                    buffer[0.5],
                                                    durations)
            detection_f1_micro = mme_res["detection"].loc["micro-avg"]["f1"]
            uniformity_f1_micro = mme_res["uniformity"].loc["micro-avg"]["f1"]
            total_duration_f1_micro = mme_res["total duration"].loc["micro-avg"]["f1"]
            relative_duration_f1_micro = mme_res["relative duration"].loc["micro-avg"]["f1"]
            detection_f1_macro = mme_res["detection"].loc["macro-avg"]["f1"]
            uniformity_f1_macro = mme_res["uniformity"].loc["macro-avg"]["f1"]
            total_duration_f1_macro = mme_res["total duration"].loc["macro-avg"]["f1"]
            relative_duration_f1_macro = mme_res["relative duration"].loc["macro-avg"]["f1"]
            '''
            self.log(f"Test/strong_SGP_{subset}_intersection_f1_macro", intersection_f1_macro)
            self.log(f"Test/strong_SGP_{subset}_event_f1_macro", event_f1_macro)
            self.log(f"Test/strong_SGP_{subset}_event_f1_micro", event_f1_micro)
            self.log(f"Test/strong_SGP_{subset}_segment_f1_macro", segment_f1_macro)
            self.log(f"Test/strong_SGP_{subset}_segment_f1_micro", segment_f1_micro)
            '''
            self.log(f"Test/strong_SGP_{subset}_detection_f1_micro", detection_f1_micro)
            self.log(f"Test/strong_SGP_{subset}_detection_f1_macro", detection_f1_macro)
            self.log(f"Test/strong_SGP_{subset}_uniformity_f1_micro", uniformity_f1_micro)
            self.log(f"Test/strong_SGP_{subset}_uniformity_f1_macro", uniformity_f1_macro)
            self.log(f"Test/strong_SGP_{subset}_total_duration_f1_micro", total_duration_f1_micro)
            self.log(f"Test/strong_SGP_{subset}_total_duration_f1_macro", total_duration_f1_macro)
            self.log(f"Test/strong_SGP_{subset}_relative_duration_f1_micro", relative_duration_f1_micro)
            self.log(f"Test/strong_SGP_{subset}_relative_duration_f1_macro", relative_duration_f1_macro)
            '''
        # save dict of metrics per instance
        self.df_metrics_sgp.to_csv(os.path.join(save_dir, "stats_SGP.csv"), index=False)
        self.df_metrics_sonyc.to_csv(os.path.join(save_dir, "stats_SONYC.csv"), index=False)

        self.reset_metrics("Test")
        self.weak_metrics["Test"]["SGP_near"].reset()    
        self.weak_metrics["Test"]["SONYC_near"].reset()
        self.weak_metrics["Test"]["SGP_far"].reset()          
        self.weak_metrics["Test"]["SONYC_far"].reset()    

    def configure_optimizers(self):
        return [self.opt]

    def train_dataloader(self):

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_data.collate_fn
        )

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=self.valid_data.collate_fn
       )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=self.test_data.collate_fn
        )
        return self.test_loader

    def update_metrics(self, split, strong_preds, weak_preds, labels, filenames, mask_SGP, mask_SONYC):
        torch.use_deterministic_algorithms(False)

        labels_weak = torch.max(labels, -1)[0].float()
        if torch.any(mask_SONYC):
            # accumulate metrics for weak labels SONYC
            self.weak_metrics[split]["SONYC_all"].update(
                weak_preds[mask_SONYC], labels_weak[mask_SONYC].int()
            )
            if split == "Test":
                self.weak_metrics_by_class["SONYC_all"].update(
                    weak_preds[mask_SONYC], labels_weak[mask_SONYC].int()
                )
                self.update_df_metrics(weak_preds[mask_SONYC], labels_weak[mask_SONYC].int(), "SONYC", "all")
            
            # accumulate metrics by weak polyphony
            polyphony = torch.sum(labels_weak[mask_SONYC] , axis=1).int()
            
            mask_monoph = (polyphony == 1).bool()
            mask_lowpolyph = torch.logical_or(polyphony  == 2, polyphony == 3).bool()
            mask_highpolyph = (polyphony > 3).bool()
            
            if torch.any(mask_monoph):
                self.weak_metrics[split]["SONYC_monoph"].update(
                    weak_preds[mask_SONYC][mask_monoph], labels_weak[mask_SONYC][mask_monoph].int()
                )
                if split == "Test":
                    self.weak_metrics_by_class["SONYC_monoph"].update(
                        weak_preds[mask_SONYC][mask_monoph], labels_weak[mask_SONYC][mask_monoph].int()
                    )          
                    self.update_df_metrics(weak_preds[mask_SONYC][mask_monoph], labels_weak[mask_SONYC][mask_monoph].int(), "SONYC", "monoph")       
                
            if torch.any(mask_lowpolyph):
                self.weak_metrics[split]["SONYC_lowpolyph"].update(
                    weak_preds[mask_SONYC][mask_lowpolyph], labels_weak[mask_SONYC][mask_lowpolyph].int()
                )
                if split == "Test":
                    self.weak_metrics_by_class["SONYC_lowpolyph"].update(
                        weak_preds[mask_SONYC][mask_lowpolyph], labels_weak[mask_SONYC][mask_lowpolyph].int()
                    )
                    self.update_df_metrics(weak_preds[mask_SONYC][mask_lowpolyph], labels_weak[mask_SONYC][mask_lowpolyph].int(), "SONYC", "lowpolyph")
            
            if torch.any(mask_highpolyph):
                self.weak_metrics[split]["SONYC_highpolyph"].update(
                    weak_preds[mask_SONYC][mask_highpolyph], labels_weak[mask_SONYC][mask_highpolyph].int()
                )
                if split == "Test":
                    self.weak_metrics_by_class["SONYC_highpolyph"].update(
                        weak_preds[mask_SONYC][mask_highpolyph], labels_weak[mask_SONYC][mask_highpolyph].int()
                    )
                    self.update_df_metrics(weak_preds[mask_SONYC][mask_highpolyph], labels_weak[mask_SONYC][mask_highpolyph].int(), "SONYC", "highpolyph")

        if torch.any(mask_SGP):                
            # accumulate metrics for weak labels SGP
            self.weak_metrics[split]["SGP_all"].update(
                weak_preds[mask_SGP], labels_weak[mask_SGP].int()
            )
            if split == "Test":
                self.weak_metrics_by_class["SGP_all"].update(
                    weak_preds[mask_SGP], labels_weak[mask_SGP].int()
                )                
                self.update_df_metrics(weak_preds[mask_SGP], labels_weak[mask_SGP].int(), "SGP", "all")
                
            '''
            # accumulate metrics by strong polyphony
            polyphony = torch.max(torch.sum(labels[mask_SGP], axis=1),axis=1)[0].int()
            
            mask_monoph = (polyphony == 1).bool()
            mask_lowpolyph = (polyphony  == 2).bool()
            mask_highpolyph = (polyphony > 2).bool()
            '''
            # accumulate metrics by weak polyphony
            polyphony = torch.sum(labels_weak[mask_SGP] , axis=1).int()
            
            mask_monoph = (polyphony == 1).bool()
            mask_lowpolyph = torch.logical_or(polyphony  == 2, polyphony == 3).bool()
            mask_highpolyph = (polyphony > 3).bool()
            
            if torch.any(mask_monoph):
                self.weak_metrics[split]["SGP_monoph"].update(
                    weak_preds[mask_SGP][mask_monoph], labels_weak[mask_SGP][mask_monoph].int()
                )
                if split == "Test":
                    self.weak_metrics_by_class["SGP_monoph"].update(
                        weak_preds[mask_SGP][mask_monoph], labels_weak[mask_SGP][mask_monoph].int()
                    )
                    self.update_df_metrics(weak_preds[mask_SGP][mask_monoph], labels_weak[mask_SGP][mask_monoph].int(), "SGP", "monoph")
                    
                    # Add preds to buffer
                    filenames_monoph = filenames[mask_SGP.cpu().numpy()][mask_monoph.cpu().numpy()]

                    decoded_student_strong = batched_decode_preds(
                        strong_preds[mask_SGP][mask_monoph],
                        filenames_monoph,
                        self.encoder,
                        median_filter=self.hparams["training"]["median_window"],
                        thresholds=list(self.strong_preds_buffers["Test_monoph"].keys()),
                    )

                    for th in self.strong_preds_buffers["Test_monoph"].keys():
                        self.strong_preds_buffers["Test_monoph"][th] = pd.concat(
                            [self.strong_preds_buffers["Test_monoph"][th], decoded_student_strong[th]]
                            , ignore_index=True
                        )
            
            if torch.any(mask_lowpolyph):
                self.weak_metrics[split]["SGP_lowpolyph"].update(
                    weak_preds[mask_SGP][mask_lowpolyph], labels_weak[mask_SGP][mask_lowpolyph].int()
                )

                if split == "Test":
                    self.weak_metrics_by_class["SGP_lowpolyph"].update(
                        weak_preds[mask_SGP][mask_lowpolyph], labels_weak[mask_SGP][mask_lowpolyph].int()
                    )
                    self.update_df_metrics(weak_preds[mask_SGP][mask_lowpolyph], labels_weak[mask_SGP][mask_lowpolyph].int(), "SGP", "lowpolyph")
                    # Add preds to buffer
                    filenames_lowpolyph = filenames[mask_SGP.cpu().numpy()][mask_lowpolyph.cpu().numpy()]

                    decoded_student_strong = batched_decode_preds(
                        strong_preds[mask_SGP][mask_lowpolyph],
                        filenames_lowpolyph,
                        self.encoder,
                        median_filter=self.hparams["training"]["median_window"],
                        thresholds=list(self.strong_preds_buffers["Test_lowpolyph"].keys()),
                    )

                    for th in self.strong_preds_buffers["Test_lowpolyph"].keys():
                        self.strong_preds_buffers["Test_lowpolyph"][th] = pd.concat(
                            [self.strong_preds_buffers["Test_lowpolyph"][th], decoded_student_strong[th]]
                            , ignore_index=True
                        )
                        
            if torch.any(mask_highpolyph):
                self.weak_metrics[split]["SGP_highpolyph"].update(
                    weak_preds[mask_SGP][mask_highpolyph], labels_weak[mask_SGP][mask_highpolyph].int()
                )
                if split == "Test":
                    self.weak_metrics_by_class["SGP_highpolyph"].update(
                                            weak_preds[mask_SGP][mask_highpolyph], labels_weak[mask_SGP][mask_highpolyph].int()
                                        )
                    self.update_df_metrics(weak_preds[mask_SGP][mask_highpolyph], labels_weak[mask_SGP][mask_highpolyph].int(), "SGP", "highpolyph")
                    # Add preds to buffer
                    filenames_highpolyph = filenames[mask_SGP.cpu().numpy()][mask_highpolyph.cpu().numpy()]

                    decoded_student_strong = batched_decode_preds(
                        strong_preds[mask_SGP][mask_highpolyph],
                        filenames_highpolyph,
                        self.encoder,
                        median_filter=self.hparams["training"]["median_window"],
                        thresholds=list(self.strong_preds_buffers["Test_highpolyph"].keys()),
                    )

                    for th in self.strong_preds_buffers["Test_highpolyph"].keys():
                        self.strong_preds_buffers["Test_highpolyph"][th] = pd.concat(
                            [self.strong_preds_buffers["Test_highpolyph"][th], decoded_student_strong[th]]
                            , ignore_index=True
                        )

            filenames_strong = [
                x
                for x in filenames
                if x[0] == '['
            ]

            decoded_student_strong = batched_decode_preds(
                strong_preds[mask_SGP],
                filenames_strong,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.strong_preds_buffers[split].keys()),
            )

            for th in self.strong_preds_buffers[split].keys():
                self.strong_preds_buffers[split][th] = pd.concat(
                    [self.strong_preds_buffers[split][th], decoded_student_strong[th]]
                    , ignore_index=True
                )

        torch.use_deterministic_algorithms(True)



    def compute_metrics(self, split, dataset, bs_SGP, bs_SONYC, save_dir=None):
        torch.use_deterministic_algorithms(False)
        obj_metric = 0        
        # compute weak metrics
        for key in self.weak_metrics[split]:
            if ("SGP" in key and bs_SGP > 0) or ("SONYC" in key and bs_SONYC > 0):
                result = self.weak_metrics[split][key].compute()
                if f"{split}/weak_SONYC_all_ap_macro" in result:
                    obj_metric += result[f"{split}/weak_SONYC_all_ap_macro"]
                if f"{split}/weak_SGP_all_ap_macro" in result:
                    obj_metric += result[f"{split}/weak_SGP_all_ap_macro"]
                self.log_dict(result)

        if split == "Test":
            for key in self.weak_metrics_by_class:
                result = self.weak_metrics_by_class[key].compute()
                result.update({"AP":torch.tensor(result["AP"])})
                result_df = pd.DataFrame(
                    {k:v.cpu().numpy() for k,v in result.items()}, index=self.encoder.taxonomy_coarse["class_labels"])
                result_df.to_csv(os.path.join(save_dir, key + ".csv"),float_format='%.2f')
        
        if bs_SGP > 0:
            # compute strong metrics
            intersection_f1_macro = compute_per_intersection_macro_f1(
                self.strong_preds_buffers[split],
                dataset[0].groundtruths,
                dataset[0].durations,
            )

            
            event_f1_macro, event_f1_micro, segment_f1_macro, segment_f1_micro = log_sedeval_metrics(
                self.strong_preds_buffers[split][0.5],
                dataset[0].groundtruths,
                save_dir,
            )
            
            
            mme_res = multi_eval.get_single_result_df(dataset[0].groundtruths,
                                                               self.strong_preds_buffers[split][0.5],
                                                               dataset[0].durations)
            
            detection_f1_micro = mme_res["detection"].loc["micro-avg"]["f1"]
            uniformity_f1_micro = mme_res["uniformity"].loc["micro-avg"]["f1"]
            total_duration_f1_micro = mme_res["total duration"].loc["micro-avg"]["f1"]
            relative_duration_f1_micro = mme_res["relative duration"].loc["micro-avg"]["f1"]
            detection_f1_macro = mme_res["detection"].loc["macro-avg"]["f1"]
            uniformity_f1_macro = mme_res["uniformity"].loc["macro-avg"]["f1"]
            total_duration_f1_macro = mme_res["total duration"].loc["macro-avg"]["f1"]
            relative_duration_f1_macro = mme_res["relative duration"].loc["macro-avg"]["f1"]
            
            
            obj_metric_strong_type = self.hparams["training"].get("obj_metric_strong_type")
            if obj_metric_strong_type is None or obj_metric_strong_type == "intersection_macro":
                strong_metric = intersection_f1_macro
            elif obj_metric_strong_type == "detection_micro":
                strong_metric = detection_f1_micro
            elif obj_metric_strong_type == "event_macro":
                strong_metric = event_f1_macro
            elif obj_metric_strong_type == "event_micro":
                strong_metric = event_f1_micro
            elif obj_metric_strong_type == "segment_macro":
                strong_metric = segment_f1_macro
            elif obj_metric_strong_type == "segment_micro":
                strong_metric = segment_f1_micro
            
            else:
                raise NotImplementedError(
                    f"obj_metric_strong_type: {obj_metric_strong_type} not implemented."
                )

            if self.hparams["training"]["batch_size"][0] > 0 and self.weight_loss_sup[0] > 0:
                obj_metric += strong_metric
            
            self.log(f"{split}/obj_metric", obj_metric)
            self.log(f"{split}/strong_SGP_intersection_f1_macro", intersection_f1_macro)
            self.log(f"{split}/strong_SGP_event_f1_macro", event_f1_macro)
            self.log(f"{split}/strong_SGP_event_f1_micro", event_f1_micro)
            self.log(f"{split}/strong_SGP_segment_f1_macro", segment_f1_macro)
            self.log(f"{split}/strong_SGP_segment_f1_micro", segment_f1_micro)
            
            self.log(f"{split}/strong_SGP_detection_f1_micro", detection_f1_micro)
            self.log(f"{split}/strong_SGP_detection_f1_macro", detection_f1_macro)
            self.log(f"{split}/strong_SGP_uniformity_f1_micro", uniformity_f1_micro)
            self.log(f"{split}/strong_SGP_uniformity_f1_macro", uniformity_f1_macro)
            self.log(f"{split}/strong_SGP_total_duration_f1_micro", total_duration_f1_micro)
            self.log(f"{split}/strong_SGP_total_duration_f1_macro", total_duration_f1_macro)
            self.log(f"{split}/strong_SGP_relative_duration_f1_micro", relative_duration_f1_micro)
            self.log(f"{split}/strong_SGP_relative_duration_f1_macro", relative_duration_f1_macro)

        torch.use_deterministic_algorithms(True)
    def _init_metrics(self):
        # for weak labels we compute F1Score, AP, AUROC, 
        num_classes = len(self.encoder.taxonomy_coarse["class_labels"])
        f1_micro = torchmetrics.F1Score(
            task='multilabel',
            num_labels=num_classes,
            average="micro",
            threshold=0.5
        )

        f1_macro = torchmetrics.F1Score(
            task='multilabel',
            num_labels=num_classes,
            average="macro",
            threshold=0.5
        )

        f1_none = torchmetrics.F1Score(
            task='multilabel',
            num_labels=num_classes,
            average="none",
            threshold=0.5
        )

        prec_micro = torchmetrics.Precision(
            task='multilabel',
            num_labels=num_classes,
            average="micro",
            threshold=0.5
        )

        prec_macro = torchmetrics.Precision(
            task='multilabel',
            num_labels=num_classes,
            average="macro",
            threshold=0.5
        )

        prec_none = torchmetrics.Precision(
            task='multilabel',
            num_labels=num_classes,
            average="none",
            threshold=0.5
        )
    
        rec_micro = torchmetrics.Recall(
            task='multilabel',
            num_labels=num_classes,
            average="micro",
            threshold=0.5
        )
        
        rec_macro = torchmetrics.Recall(
            task='multilabel',
            num_labels=num_classes,
            average="macro",
            threshold=0.5
        )
        
        rec_none = torchmetrics.Recall(
            task='multilabel',
            num_labels=num_classes,
            average="none",
            threshold=0.5
        )

        ap_micro = torchmetrics.AveragePrecision(
            task='multilabel',
            num_labels=num_classes,
            average="micro",
        )

        ap_macro = torchmetrics.AveragePrecision(
            task='multilabel',
            num_labels=num_classes,
            average="macro",
        )
        
        ap_none = torchmetrics.AveragePrecision(
            task='multilabel',
            num_labels=num_classes,
            average="none",
        )

        label_ranking = torchmetrics.classification.MultilabelRankingAveragePrecision(num_labels=num_classes)
        
        weak_metrics = torchmetrics.MetricCollection(
            {
                'precision_micro': prec_micro,
                'precision_macro': prec_macro,
                'recall_micro' : rec_micro,
                'recall_macro' : rec_macro,
                'F1_score_micro': f1_micro,
                'F1_score_macro':f1_macro,
                'ap_micro':ap_micro,
                'ap_macro':ap_macro,
                'label_ranking':label_ranking,
            }
            
        )

        weak_metrics_by_class = torchmetrics.MetricCollection(
            {
                'Precision': prec_none,
                'Recall' : rec_none,
                'F1 score': f1_none,
                'AP':ap_none,
            }
            
        ) 
        
        # store each torchmetrics object specific to a data split and subset (SONYC, SGP, SONYC_monophonic ...) in a dictionary
        
        self.weak_metrics = torch.nn.ModuleDict({"Train":torch.nn.ModuleDict(), "Val":torch.nn.ModuleDict(), "Test":torch.nn.ModuleDict()})
        
        for split in ["Train", "Val", "Test"]:
            self.weak_metrics[split]["SGP_all"] = weak_metrics.clone(prefix=f"{split}/weak_SGP_all_")        
            self.weak_metrics[split]["SONYC_all"] = weak_metrics.clone(prefix=f"{split}/weak_SONYC_all_")
            self.weak_metrics[split]["SGP_monoph"] = weak_metrics.clone(prefix=f"{split}/weak_SGP_monoph_")
            self.weak_metrics[split]["SONYC_monoph"] = weak_metrics.clone(prefix=f"{split}/weak_SONYC_monoph_")
            self.weak_metrics[split]["SGP_lowpolyph"] = weak_metrics.clone(prefix=f"{split}/weak_SGP_lowpolyph_")
            self.weak_metrics[split]["SONYC_lowpolyph"] = weak_metrics.clone(prefix=f"{split}/weak_SONYC_lowpolyph_")
            self.weak_metrics[split]["SGP_highpolyph"] = weak_metrics.clone(prefix=f"{split}/weak_SGP_highpolyph_")
            self.weak_metrics[split]["SONYC_highpolyph"] = weak_metrics.clone(prefix=f"{split}/weak_SONYC_highpolyph_")
        
        self.weak_metrics["Test"]["SGP_near"] = weak_metrics.clone(prefix=f"Test/weak_SGP_near_")            
        self.weak_metrics["Test"]["SONYC_near"] = weak_metrics.clone(prefix=f"Test/weak_SONYC_near_")
        self.weak_metrics["Test"]["SGP_far"] = weak_metrics.clone(prefix=f"Test/weak_SGP_far_")            
        self.weak_metrics["Test"]["SONYC_far"] = weak_metrics.clone(prefix=f"Test/weak_SONYC_far_")
        
        self.weak_metrics_by_class = torch.nn.ModuleDict()
        self.weak_metrics_by_class["SGP_all"] = weak_metrics_by_class.clone()        
        self.weak_metrics_by_class["SONYC_all"] = weak_metrics_by_class.clone()
        self.weak_metrics_by_class["SGP_monoph"] = weak_metrics_by_class.clone()
        self.weak_metrics_by_class["SONYC_monoph"] = weak_metrics_by_class.clone()
        self.weak_metrics_by_class["SGP_lowpolyph"] = weak_metrics_by_class.clone()
        self.weak_metrics_by_class["SONYC_lowpolyph"] = weak_metrics_by_class.clone()
        self.weak_metrics_by_class["SGP_highpolyph"] = weak_metrics_by_class.clone()
        self.weak_metrics_by_class["SONYC_highpolyph"] = weak_metrics_by_class.clone()
        self.weak_metrics_by_class["SGP_near"] = weak_metrics_by_class.clone()        
        self.weak_metrics_by_class["SGP_far"] = weak_metrics_by_class.clone()        
        self.weak_metrics_by_class["SONYC_near"] = weak_metrics_by_class.clone()        
        self.weak_metrics_by_class["SONYC_far"] = weak_metrics_by_class.clone()        
    
    
        # buffer for event based scores which we compute using sed-eval
        self.sed_buffer = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        
        self.strong_preds_buffers = {split:deepcopy(self.sed_buffer) for split in ["Train", "Val", "Test", "Test_near", "Test_far", "Test_monoph", "Test_lowpolyph","Test_highpolyph"]}
        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        test_thresholds =  np.arange(1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds)
        self.strong_preds_buffers["test_psds"] = {k: pd.DataFrame() for k in test_thresholds}

        # Dictionnary for sample-level statistics of our metrics
        self.df_metrics_sonyc = pd.DataFrame(columns=["name", "subset", "ap", "lrap", "f1","precision", "recall"])
        self.df_metrics_sgp = pd.DataFrame(columns=["name", "subset", "ap", "lrap", "f1","precision", "recall"])

    
    def reset_metrics(self,split):
        for key in self.weak_metrics[split]:
            self.weak_metrics[split][key].reset()
        
        self.strong_preds_buffers[split] = deepcopy(self.sed_buffer)
        
    def update_df_metrics(self, pred, target, dset, subset):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        rows = []
        n_samples = 4
        for i in range(1+len(pred)//n_samples):
            if i*n_samples >= len(pred):
                continue
            t = target[i*n_samples:(i+1)*n_samples]
            p = pred[i*n_samples:(i+1)*n_samples]
            p05 = (p>0.5).astype(int)
            ap = M.average_precision_score(t, p, average='samples')
            lrap = M.label_ranking_average_precision_score(t, p)
            f1 = M.f1_score(t, p05, average='samples')
            precision = M.precision_score(t, p05, average='samples', zero_division=0)
            recall = M.recall_score(t, p05, average='samples')
            
            rows.append({"name": self.hparams["experiment_name"],
                        "subset": subset,
                        "ap": ap,
                        "lrap": lrap,
                        "f1": f1,
                        "precision": precision,
                        "recall" : recall,})
        
        if dset == "SONYC":
            self.df_metrics_sonyc = pd.concat([self.df_metrics_sonyc, pd.DataFrame.from_records(rows)])
        elif dset == "SGP":
            self.df_metrics_sgp =  pd.concat([self.df_metrics_sgp, pd.DataFrame.from_records(rows)])