import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
import itertools

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)


def aggr_events(events_raw):
    """Aggregates event vector to the list of compact event vectors.
    Parameters:
        events_raw  --  vector of raw events
    Returns:
        events_aggr --  list of compact event vectors ([onset, offset, event])
    """

    events_aggr = []
    s = 0
    for bit, group in itertools.groupby(events_raw):
        event_length = len(list(group))
        e = s + event_length
        events_aggr.append([s, e, bit])
        s = e
    return events_aggr


def calc_evt(data, fs=1000, fast=False):
    """Calculated event data"""
    evt_compact = aggr_events(data["evt"])
    evt = pd.DataFrame(evt_compact, columns=["s", "e", "evt"])
    evt["dur_s"] = np.diff(evt[["s", "e"]], axis=1).squeeze()
    evt["dur"] = evt["dur_s"] / fs
    return evt


class job:
    def __init__(
        self,
        matchers: dict = None,
        multiclass_strategy: list = None,
        binary_strategy: list = None,
        event_map: dict = None,
        label: str = None,
    ):
        self.matchers = matchers or {
            "maximum-overlap": {},
            "iou": {},
            "iou/05": {"iou_threshold": 0.5},
            "sample": {},
        }
        self.multiclass_strategy = multiclass_strategy or [
            "all",
            "ignore_matched_undef",
            "ignore_unmatched_undef",
            "ignore_undef",
        ]
        self.binary_strategy = binary_strategy or ["tn", "error", "ignore"]
        self.event_map = event_map or {
            "1": 1,
            "2": 2,
            "4": 4,
            "0": 100,
        }  # TODO: This is wrong, for the lookAtPoint Dataset. Change to correct values
        self.label = label or "default_label"


class pp_args:
    def __init__(
        self,
        sacc_minimum_distance: float = 0.1,
        pp_kwargs: dict = {
            "thres_id": 75.0,
            "thres_ifa": 0.2,
            "thres_ifi": 75.0,
            "thres_sd_s": 3,
            "thres_pd": 3,
            "thres_isi": 25.0,
            "thres_sd_lo": 6.0,
            "thres_sd_hi": 150.0,
            "thres_fd": 50.0,
        },
    ):
        self.sacc_minimum_distance = sacc_minimum_distance
        self.pp_kwargs = pp_kwargs


class ETData:
    # Data types and constants
    dtype = np.dtype(
        [
            ("t", np.float64),
            ("x", np.float32),
            ("y", np.float32),
            ("status", np.bool_),
            ("evt", np.uint8),
        ]
    )
    evt_color_map = dict(
        {
            0: "gray",  # 0. Undefined
            1: "b",  # 1. Fixation
            2: "r",  # 2. Saccade
            3: "y",  # 3. Post-saccadic oscillation
            4: "m",  # 4. Smooth pursuit
            5: "k",  # 5. Blink
        }
    )

    def __init__(self, df):
        self.data = np.array(df.to_records(index=False), dtype=ETData.dtype)
        self.fs = self.find_nearest_fs(self.data["t"])
        self.evt = None

    def load(self, fpath, **kwargs):
        """Loads data.
        Parameters:
            fpath   --  file path
            kwargs:
                'source'. Available values:
                          'etdata'    --  numpy array with ETData.dtype
                          function    --  function, which parses custom
                                          data format and returns numpy array,
                                          which can be converted to have data
                                          type of ETData.dtype
        """

        if not ("source" in kwargs):
            try:
                self.data = np.load(fpath)
            except:
                print(("ERROR loading %s" % fpath))
        else:
            if kwargs["source"] == "etdata":
                self.data = np.load(fpath)

            if kwargs["source"] == "array":
                self.data = fpath

            if callable(kwargs["source"]):
                self.data = kwargs["source"](fpath, ETData.dtype)

        # estimate sampling rate
        self.fs = float(self.find_nearest_fs(self.data["t"]))
        self.evt = None
        return self.data

    def save(self, spath):
        """Saves data as numpy array with ETData.dtype data type.
        Parameters:
            spath   --  save path
        """
        np.save(spath, self.data)

    def find_nearest_fs(self, t):
        """Estimates data sampling frequency.
        Parameters:
            t   --  timestamp vector
        Returns:
            Estimated sampling frequency
        """
        fs = np.array(
            [
                2000,
                1250,
                1000,
                600,
                500,  # high end
                300,
                250,
                240,
                200,  # middle end
                120,
                75,
                60,
                50,
                30,
                25,
            ]
        )  # low end
        t = np.median(1 / np.diff(t))
        return fs.flat[np.abs(fs - t).argmin()]

    def calc_evt(self, fast=False):
        """Calculated event data"""
        evt_compact = aggr_events(self.data["evt"])
        evt = pd.DataFrame(evt_compact, columns=["s", "e", "evt"])
        evt["dur_s"] = np.diff(evt[["s", "e"]], axis=1).squeeze()
        evt["dur"] = evt["dur_s"] / self.fs

        if not (fast):
            (
                evt["posx_s"],
                evt["posx_e"],
                evt["posy_s"],
                evt["posy_e"],
                evt["posx_mean"],
                evt["posy_mean"],
                evt["posx_med"],
                evt["posy_med"],
                evt["pv"],
                evt["pv_index"],
                evt["rms"],
                evt["std"],
            ) = list(zip(*[calc_event_data(self, x) for x in evt_compact]))
            evt["ampl_x"] = np.diff(evt[["posx_s", "posx_e"]])
            evt["ampl_y"] = np.diff(evt[["posy_s", "posy_e"]])
            evt["ampl"] = np.hypot(evt["ampl_x"], evt["ampl_y"])
        # TODO:
        #   calculate fix-to-fix saccade amplitude
        self.evt = evt
        return self.evt
    
    def shape(self):
        return self.data.shape

    def plot(self, spath=None, save=False, show=True, title=None):
        """Plots trial"""
        if show:
            plt.ion()
        else:
            plt.ioff()

        fig = plt.figure(figsize=(10, 6))
        ax00 = plt.subplot2grid((2, 2), (0, 0))
        ax10 = plt.subplot2grid((2, 2), (1, 0), sharex=ax00)
        ax01 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

        ax00.plot(self.data["t"], self.data["x"], "-")
        ax10.plot(self.data["t"], self.data["y"], "-")
        ax01.plot(self.data["x"], self.data["y"], "-")
        for e, c in ETData.evt_color_map.items():
            mask = self.data["evt"] == e
            ax00.plot(self.data["t"][mask], self.data["x"][mask], ".", color=c)
            ax10.plot(self.data["t"][mask], self.data["y"][mask], ".", color=c)
            ax01.plot(self.data["x"][mask], self.data["y"][mask], ".", color=c)

        etdata_extent = np.nanmax([np.abs(self.data["x"]), np.abs(self.data["y"])]) + 1

        ax00.axis([self.data["t"].min(), self.data["t"].max(), -etdata_extent, etdata_extent])
        ax10.axis([self.data["t"].min(), self.data["t"].max(), -etdata_extent, etdata_extent])
        ax01.axis([-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])

        #        sns.despine()
        if title is not None:
            plt.suptitle(title)
        plt.tight_layout()

        if save and not (spath is None):
            plt.savefig("%s.png" % (spath))
            plt.close()
