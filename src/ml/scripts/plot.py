import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from jsonargparse import CLI, ActionConfigFile, ArgumentParser

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)

import eval.misc.utils as utils
from eval.misc import eval_utils, matching

from ml.utils.classes import job


def main():
    parser = ArgumentParser()
    parser.add_class_arguments(job, "jobs")
    parser.add_argument("-c", "--config", action=ActionConfigFile)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="/home/martin/Documents/Exjobb/eed/.experiments/plots",
        help="Output directory",
    )
    parser.add_argument(
        "-s",
        "--save-results",
        action="store_true",
        help="Save results to csv file",
    )
    args = parser.parse_args()
    save_results = args.save_results
    jpath = ""
    # Define the path to the predictions and ground truths files
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    odir = args.output
    fpath_pr = ".experiments/results/sklearn/validation/lookAtPoint_EL_S5_pd.csv"  # ".experiments/results/sklearn/validation/lookAtPoint_EL_S5_pd.csv"
    fpath_gt = ".experiments/results/sklearn/validation/lookAtPoint_EL_S5_gt.csv"  # ".experiments/results/sklearn/validation/lookAtPoint_EL_S5_gt.csv"

    # Get the filename
    _pr = os.path.splitext(os.path.basename(fpath_pr))[0]
    _gt = os.path.splitext(os.path.basename(fpath_gt))[0]
    fname = f"{_pr}_vs_{_gt}"
    # Check if the files exist
    if not os.path.exists(fpath_pr) or not os.path.exists(fpath_gt):
        print("Error: Predictions or ground truths file not found.")
        exit()
    jobs = args.jobs
    matchers = jobs.matchers
    job_label = jobs.label
    multiclass_strategy = jobs.multiclass_strategy
    binary_strategy = jobs.binary_strategy
    event_map = jobs.event_map
    event_map = utils.keys2num(event_map)

    event_labels = event_map.values()
    if 0 not in event_map.keys():
        # add undef label
        event_labels = [0, *event_labels]

    data_gt, data_pr = utils.load_data(fpath_gt, fpath_pr, event_map)
    event_labels = list(set(event_labels))
    event_matcher = matching.EventMatcher(gt=data_gt, pr=data_pr)
    meta_accum = []
    meta = {"gt": _gt, "pr": _pr, "fname": fname}

    # run eval
    result_accum = []
    for matcher, matching_kwargs in matchers.items():
        matcher_label = filter(None, (matcher, job_label))
        matcher_label = {"matcher": "-".join(matcher_label)}

        match_plot_kwargs = jobs.get("match-plot-kwargs", None)
        kwargs = matching_kwargs if isinstance(matching_kwargs, list) else [matching_kwargs]
        # add plot mode indicator
        _plot_mode = {"plot-mode": True}
        kwargs = [utils.merge_dicts([_kwargs, _plot_mode]) for _kwargs in kwargs]

        # run matching
        _match_result = [event_matcher.run_matching(matcher, **_kwargs) for _kwargs in kwargs]
        _, events = zip(*_match_result)


        eval_result = eval_utils.calc_scores(
                    event_matcher=event_matcher,
                    matcher=matcher,
                    matching_kwargs=matching_kwargs,
                    labels=event_labels,
                    multiclass_strategy=multiclass_strategy,
                    binary_strategy=binary_strategy,
                    meta=[meta, matcher_label],
                    unittest=None,
                )
        result_accum.extend(eval_result)
        result = pd.DataFrame(result_accum)
        iou_mcc = result[result["eval"] == "multiclass"]["mcc"]
        iou_accuracy = result[result["eval"] == "multiclass"]["accuracy"]
        iou_accuracy_balanced = result[result["eval"] == "multiclass"]["accuracy_balanced"]
        iou_kappa = result[result["eval"] == "multiclass"]["kappa"]
        iou_nld = result[result["eval"] == "multiclass"]["nld"]

        print("iou_accuracy: ", iou_accuracy.values)
        print("iou_accuracy_balanced: ", iou_accuracy_balanced.values)
        print("iou_kappa: ", iou_kappa.values)
        print("iou_nld: ", iou_nld.values)
        print("iou_mcc: ", iou_mcc.values)
        # interactive plot
        utils.plot_job(
            matcher=event_matcher,
            events=events,
            spath=os.path.relpath(fpath_gt, root),
            odir=odir,
            match_plot_kwargs=match_plot_kwargs,
            matcher_label=matcher_label["matcher"],
            data=data_gt,

        )
    


    result = pd.DataFrame(result_accum)
    iou_mcc = result[result["eval"] == "multiclass"]["mcc"]
    print(result[result["eval"] == "multiclass"])
    print(iou_mcc)

    if save_results:
        _, jname = utils.split_path(jpath)
        rpath = os.path.join(odir, f"{jname}.csv")
        result.to_csv(rpath, index=False)
        result_agg = result.groupby(["gt", "pr", "matcher", "eval", "event"], as_index=False).mean(numeric_only=True)

        result_agg.to_csv(os.path.join(odir, f"{jname}-agg.csv"), index=False)

if __name__ == "__main__":
    main()
