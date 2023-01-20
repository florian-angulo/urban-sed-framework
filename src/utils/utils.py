import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from .evaluation_measures import compute_sed_eval_metrics


def nantensor(*args, **kwargs):
    return torch.ones(*args, **kwargs) * np.nan


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

def batched_decode_preds(
    strong_preds, filenames, encoder, thresholds=[0.5], median_filter=7, pad_idx=None,
):
    """Decode a batch of predictions to dataframes. Each threshold gives a different dataframe
    and is stored in a dictionary

    Args:
        strong_preds (torch.Tensor): batch of strong predictions
        filenames (list): list of filenames of the current batch
        encoder (ManyHotEncoder): object used to decode predictions
        thresholds (list, optional): list of thresholds. Defaults to [0.5].
        median_filter (int, optional): median smoothing window size (in frames). Defaults to 7.
        pad_idx ([type], optional): list of indices which have ben used for padding. Defaults to None.
    
    Returns:
        dict of predictions, each keys is a threshold and the value is the DataFrame of predictions.
    """
    
    # Init a dataframe per threshold
    prediction_dfs = {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()
        
    for j in range(strong_preds.shape[0]): # over batches
        for c_th in thresholds:
            c_preds = strong_preds[j]
            if pad_idx is not None:
                true_len = int(c_preds.shape[-1] * pad_idx[j].item())
                c_preds = c_preds[:true_len]
            pred = c_preds.detach().cpu().numpy()
            pred = pred > c_th
            pred = scipy.ndimage.median_filter(pred, (1, median_filter))
            pred = encoder.decode_strong(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = Path(filenames[j]).stem
            prediction_dfs[c_th] = pd.concat([prediction_dfs[c_th],pred], ignore_index=True)
            
    return prediction_dfs


def convert_to_event_based(weak_dataframe):
    """Convert a weakly labeled DataFrame ('filename', 'event_labels') to a DataFrame strongly labeled
    ('filename', 'onset', 'offset', 'event_label')

    Args:
        weak_dataframe (pd.DataFrame) : the dataframe to be converted
    Returns:
        pd.DataFrame, the dataframe strongly labeled
    """
    
    new = []
    for _, r in weak_dataframe.iterrows():
        
        events = r["events_labels"].split(",")
        for e in events:
            new.append(
                {"filename": r["filename"], "event_label": e, "onset":0, "offset": 10}
            )
    return pd.DataFrame(new)


def log_sedeval_metrics(predictions, gt, save_dir=None):
    """Return the set of metrics from sed_eval

    Args:
        predictions (pd.DataFrame): dataframe of predictions
        gt (pd.DataFrame): dataframe of groundtruths
        save_dir ([type], optional): [description]. Defaults to None.
        
    Returns:
        tuple, event-based macro-F1 and micro-F1, segment-based macro-F1 and micro-F1
    """
    if predictions.empty:
        return 0.0, 0.0, 0.0, 0.0
    
    event_res, segment_res = compute_sed_eval_metrics(predictions, gt)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
            f.write(str(event_res))
        
        with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
            f.write(str(segment_res))
        
    return (
        event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        event_res.results()["overall"]["f_measure"]["f_measure"],
        segment_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        segment_res.results()["overall"]["f_measure"]["f_measure"]
    )   # return also segment measures

def get_rescale_matrix(encoder_in, encoder_out):
    """
    This function is used to adapt the time resolution of the one-hot labels stored in the hdf5 files.
    Used when the time resolution of the encoder output (CNN or HEAR encoder) is different than 156 for 10s audio.
    """
    
    if encoder_in.n_frames == encoder_out.n_frames:
        return np.eye((encoder_in.n_frames, encoder_out.n_frames))
    elif encoder_in.n_frames >= encoder_out.n_frames:
        coarse_encoder = encoder_out
        fine_encoder = encoder_in
        transpose = True
    else:
        fine_encoder = encoder_out
        coarse_encoder = encoder_in
        transpose = False
    
    r_matrix = torch.zeros(coarse_encoder.n_frames, fine_encoder.n_frames)
    for i in range(r_matrix.shape[1]):
        lookup = min(coarse_encoder.n_frames-1,int(coarse_encoder._time_to_frame(fine_encoder._frame_to_time(i))))
        r_matrix[lookup, i] = 1
    
    
    
    return r_matrix.T if transpose else r_matrix



def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2,
    reduction: str = "mean",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss