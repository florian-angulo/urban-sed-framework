import os

import numpy as np
import pandas as pd
import psds_eval
import sed_eval
from psds_eval import PSDSEval, plot_psd_roc, plot_per_class_psd_roc


def get_event_list_current_file(df, fname):
    """
    Get list of events for a given filename
    Args:
        df: pd.DataFrame, the dataframe to search on
        fname: the filename to extract the value from the dataframe
    Returns:
         list of events (dictionaries) for the given filename
    """
    event_file = df[df["filename"] == fname]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict("records")
    else:
        event_list_for_current_file = event_file.to_dict("records")

    return event_list_for_current_file


def psds_results(psds_obj):
    """ Compute psds scores
    Args:
        psds_obj: psds_eval.PSDSEval object with operating points.
    Returns:
    """
    try:
        psds_score = psds_obj.psds(alpha_ct=0, alpha_st=0, max_efpr=100)
        print(f"\nPSD-Score (0, 0, 100): {psds_score.value:.5f}")
        psds_score = psds_obj.psds(alpha_ct=1, alpha_st=0, max_efpr=100)
        print(f"\nPSD-Score (1, 0, 100): {psds_score.value:.5f}")
        psds_score = psds_obj.psds(alpha_ct=0, alpha_st=1, max_efpr=100)
        print(f"\nPSD-Score (0, 1, 100): {psds_score.value:.5f}")
    except psds_eval.psds.PSDSEvalError as e:
        print("psds did not work ....")
        raise EnvironmentError


def event_based_evaluation_df(
    reference, estimated, t_collar=0.200, percentage_of_length=0.2
):
    """ Calculate EventBasedMetric given a reference and estimated dataframe

    Args:
        reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            reference events
        estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            estimated events to be compared with reference
        t_collar: float, in seconds, the number of time allowed on onsets and offsets
        percentage_of_length: float, between 0 and 1, the percentage of length of the file allowed on the offset
    Returns:
         sed_eval.sound_event.EventBasedMetrics with the scores
    """

    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling="zero_score",
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return event_based_metric


def segment_based_evaluation_df(reference, estimated, time_resolution=1.0):
    """ Calculate SegmentBasedMetrics given a reference and estimated dataframe

        Args:
            reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
                reference events
            estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
                estimated events to be compared with reference
            time_resolution: float, the time resolution of the segment based metric
        Returns:
             sed_eval.sound_event.SegmentBasedMetrics with the scores
        """
    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes, time_resolution=time_resolution
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        segment_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return segment_based_metric


def compute_sed_eval_metrics(predictions, groundtruth):
    """ Compute sed_eval metrics event based and segment based with default parameters used in the task.
    Args:
        predictions: pd.DataFrame, predictions dataframe
        groundtruth: pd.DataFrame, groundtruth dataframe
    Returns:
        tuple, (sed_eval.sound_event.EventBasedMetrics, sed_eval.sound_event.SegmentBasedMetrics)
    """
    metric_event = event_based_evaluation_df(
        groundtruth, predictions, t_collar=0.200, percentage_of_length=0.2
    )
    metric_segment = segment_based_evaluation_df(
        groundtruth, predictions, time_resolution=1.0
    )

    return metric_event, metric_segment


def compute_per_intersection_macro_f1(
    prediction_dfs,
    gt,
    durations,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
):
    """ Compute F1-score per intersection, using the defautl
    Args:
        prediction_dfs: dict, a dictionary with thresholds keys and predictions dataframe
        ground_truth_file: pd.DataFrame, the groundtruth dataframe
        durations_file: pd.DataFrame, the duration dataframe
        dtc_threshold: float, the parameter used in PSDSEval, percentage of tolerance for groundtruth intersection
            with predictions
        gtc_threshold: float, the parameter used in PSDSEval percentage of tolerance for predictions intersection
            with groundtruth
        gtc_threshold: float, the parameter used in PSDSEval to know the percentage needed to count FP as cross-trigger

    Returns:

    """

    psds = PSDSEval(
        ground_truth=gt,
        metadata=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
    )
    psds_macro_f1 = []
    for threshold in prediction_dfs.keys():
        if not prediction_dfs[threshold].empty:
            threshold_f1, _ = psds.compute_macro_f_score(prediction_dfs[threshold])
        else:
            threshold_f1 = 0
        if np.isnan(threshold_f1):
            threshold_f1 = 0.0
        psds_macro_f1.append(threshold_f1)
    psds_macro_f1 = np.mean(psds_macro_f1)
    return psds_macro_f1


def compute_psds_from_operating_points(
    prediction_dfs,
    gt,
    durations,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
    alpha_ct=0,
    alpha_st=0,
    max_efpr=100,
    save_dir=None,
):

    psds_eval = PSDSEval(
        ground_truth=gt,
        metadata=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
    )

    for i, k in enumerate(prediction_dfs.keys()):
        det = prediction_dfs[k]
        # see issue https://github.com/audioanalytic/psds_eval/issues/3
        det["index"] = range(1, len(det) + 1)
        det = det.set_index("index")
        psds_eval.add_operating_point(
            det, info={"name": f"Op {i + 1:02d}", "threshold": k}
        )

    psds_score = psds_eval.psds(alpha_ct=alpha_ct, alpha_st=alpha_st, max_efpr=max_efpr)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        pred_dir = os.path.join(
            save_dir,
            f"predictions_dtc{dtc_threshold}_gtc{gtc_threshold}_cttc{cttc_threshold}",
        )
        os.makedirs(pred_dir, exist_ok=True)
        for k in prediction_dfs.keys():
            prediction_dfs[k].to_csv(
                os.path.join(pred_dir, f"predictions_th_{k:.2f}.tsv"),
                sep="\t",
                index=False,
            )

        plot_psd_roc(
            psds_score,
            filename=os.path.join(save_dir, f"PSDS_ct{alpha_ct}_st{alpha_st}_100.png"),
        )
        
        class_names = psds_eval.class_names

        tpr_vs_fpr_lin, _, _ = psds_eval.psd_roc_curves(alpha_ct=0.)
        plot_per_class_psd_roc(tpr_vs_fpr_lin, class_names, filename=os.path.join(save_dir, f"per-class TPR vs FPR PSDROC.png"), title="per-class TPR vs FPR PSDROC", xlabel="FPR", xlim=100)
        
    
    # Recover some of the operating points for each class based on f1-score
    class_constraints = list()
    # find the op. point with maximum f1-score for the fourth class
    for c_name in psds_eval.class_names:
        if c_name == "injected_psds_world_label":
            continue
        class_constraints.append({"class_name": c_name,
                                "constraint": "fscore",
                                "value": None})
    class_constraints_table = pd.DataFrame(class_constraints)
    selected_ops = psds_eval.select_operating_points_per_class(
        class_constraints_table, alpha_ct=1., beta=1.)

    with open(os.path.join(save_dir, f"best_operating_points_per_class(PSDS_ct{alpha_ct}_st{alpha_st}_100).txt"), 'w') as textfile:
        for k in range(len(class_constraints)):
            textfile.writelines([
                f"\n\nFor class {class_constraints_table.class_name[k]}, the best "
                f"op. point with {class_constraints_table.constraint[k]} ~ "
                f"{class_constraints_table.value[k]}:"
                ])
            textfile.writelines([
                f"\tProbability Threshold: {selected_ops.threshold[k]}, "
                f"TPR: {selected_ops.TPR[k]:.2f}, "
                f"FPR: {selected_ops.FPR[k]:.2f}, "
                f"eFPR: {selected_ops.eFPR[k]:.2f}, "
                f"F1-score: {selected_ops.Fscore[k]:.2f}"
                ])
    
    return psds_score.value
