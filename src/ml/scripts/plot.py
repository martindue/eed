import os, sys

import matplotlib.pyplot as plt
import pandas as pd
from jsonargparse import ActionConfigFile, CLI, ArgumentParser


project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)

from eval.misc import matching
from eval.misc import eval_utils
from ml.utils.classes import job
import eval.misc.utils as utils


def main():
    parser = ArgumentParser()
    parser.add_class_arguments(job, "jobs")
    parser.add_argument("-c", "--config", action=ActionConfigFile)
    parser.add_argument("-o", "--output", type=str, default="/home/martin/Documents/Exjobb/eed/.experiments/plots", help="Output directory")
    args = parser.parse_args()

    # Define the path to the predictions and ground truths files
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", ".."))
    odir = args.output
    fpath_pr = '.experiments/results/sklearn/train/lookAtPoint_EL_S1_pd.csv'
    fpath_gt = '.experiments/results/sklearn/train/lookAtPoint_EL_S1_gt.csv'
    
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
    # run eval
    for matcher, matching_kwargs in matchers.items():
        matcher_label = filter(None, (matcher, job_label))
        matcher_label = {"matcher": "-".join(matcher_label)}


        match_plot_kwargs = jobs.get("match-plot-kwargs", None)
        kwargs = (
            matching_kwargs
            if isinstance(matching_kwargs, list)
            else [matching_kwargs]
        )
        # add plot mode indicator
        _plot_mode = {"plot-mode": True}
        kwargs = [
            utils.merge_dicts([_kwargs, _plot_mode]) for _kwargs in kwargs
        ]

        # run matching
        _match_result = [
            event_matcher.run_matching(matcher, **_kwargs)
            for _kwargs in kwargs
        ]
        _, events = zip(*_match_result)

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
        continue


if __name__ == "__main__":
    main()
