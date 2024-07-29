import pandas as pd
import os, sys, copy
import itertools
import numpy as np
import re

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)


from ml.utils.classes import ETData

from ml.utils.helpers import round_up


class hpp:
    """Implements hard post-processing of event data"""

    def __init__(self):
        self.check_accum = {
            "short_saccades": 0,
            "long_saccades": 0,
            "saccades": 0,
            "sacc202": 0,  # saccades surrouding undef
            "sacc20": 0,  # saccade before undef
            "sacc02": 0,  # saccade after undef
            "sacc_isi": 0,
            "short_fixations": 0,
            "fixations": 0,
            "short_pso": 0,
            "pso": 0,
            "pso23": 0,  # proper pso
            "pso13": 0,  # pso after fixation
            "pso03": 0,  # pso after undefined
        }

        self.check_inds = {
            "short_saccades": 0,
            "long_saccades": 0,
            "saccades": 0,
            "sacc202": 0,  # saccades surrouding undef
            "sacc20": 0,  # saccade before undef
            "sacc02": 0,  # saccade after undef
            "sacc_isi": 0,
            "short_fixations": 0,
            "fixations": 0,
            "short_pso": 0,
            "pso": 0,
            "pso23": 0,  # proper pso
            "pso13": 0,  # pso after fixation
            "pso03": 0,  # pso after undefined
        }

    def reset_accum(self):
        """Resets check accumulator"""
        for k, v in self.check_accum.items():
            self.check_accum[k] = 0

    def run_pp(self, etdata, pp=True, **kwargs):
        """Performs post-processing sanity check and hard-removes or replaces events
        Parameters:
            etdata  --  instance of ETData()
            pp      --  if True, post-processing is performed. Otherwise only counts cases
        Returns:
            Numpy array of updated event stream
            Numpy array of updated status
            Dictionary with event status accumulator
        """

        _evt = etdata.calc_evt(fast=True)
        status = etdata.data["status"]
        fs = etdata.fs

        check = self.check_accum

        # %%pp
        ### Saccade sanity check
        _sacc = _evt.query("evt==2")
        check["saccades"] += len(_sacc)

        # check isi
        _isi = (_sacc[1:]["s"].values - _sacc[:-1]["e"].values) / fs
        _isi_inds = np.where(_isi < kwargs["thres_isi"] / 1000.0)[0]
        check["sacc_isi"] += len(_isi_inds)
        self.check_inds["sacc_isi"] = _sacc.index[_isi_inds].values

        # TODO: implement isi merging
        #        if pp:
        #
        #            _etdata = copy.deepcopy(etdata)
        #            _evt_unfold = [_e for _, e in _evt.iterrows()
        #                          for _e in itertools.repeat(e['evt'],
        #                                                     int(np.diff(e[['s', 'e']])))]
        #
        #        _etdata.data['evt'] = _evt_unfold
        #        _etdata.calc_evt(fast=True)
        #        _evt = _etdata.evt

        # pp: remove short saccades
        #        _sdur_thres = max([0.006, float(3/etdata.fs)])
        #        _sdur = _evt.query('evt==2 and dur<@_sdur_thres')

        thres_sd_lo = kwargs["thres_sd_lo"] / 1000.0
        thres_sd_lo_s = round_up(thres_sd_lo * fs,2)

        _sdur = _evt.query("evt==2 and (dur<@thres_sd_lo or dur_s<@thres_sd_lo_s)")
        check["short_saccades"] += len(_sdur)
        self.check_inds["short_saccades"] = _sdur.index.values
        if pp:
            _evt.loc[_sdur.index, "evt"] = 0

        # check long saccades.
        thres_sd_hi = kwargs["thres_sd_hi"] / 1000.0
        thres_sd_hi_s = round_up(thres_sd_hi * fs)
        _sdur = _evt.query("evt==2 and dur_s>@thres_sd_hi_s")
        check["long_saccades"] += len(_sdur)
        self.check_inds["long_saccades"] = _sdur.index.values
        if pp:
            _evt.loc[_sdur.index, "evt"] = 0

        # pp: find saccades surrounding undef;
        _sacc_check = {"202": 0, "20": 0, "02": 0}
        seq = "".join(map(str, _evt["evt"]))
        for pattern in list(_sacc_check.keys()):
            _check = np.array([m.start() for m in re.finditer("(?=%s)" % pattern, seq)])
            if not (len(_check)):
                continue

            _sacc_check[pattern] += len(_check)
            self.check_inds["sacc%s" % pattern] = _check
        #            #pp: remove saccades surrounding undef; not used anymore
        #            if pp:
        #                if (pattern=='202'):
        #                    assert ((_evt.loc[_check+1, 'evt']==0).all() and
        #                            (_evt.loc[_check+2, 'evt']==2).all())
        #                    _evt.loc[_check, 'evt'] = 0
        #                    _evt.loc[_check+2, 'evt'] = 0
        #
        ##                if (pattern=='20'):
        ##                    assert (_evt.loc[_check+1, 'evt']==0).all()
        ##                    _evt.loc[_check, 'evt'] = 0
        ##                if (pattern=='02'):
        ##                    assert (_evt.loc[_check+1, 'evt']==2).all()
        ##                    _evt.loc[_check+1, 'evt'] = 0
        #                seq=''.join(map(str, _evt['evt']))

        check["sacc202"] += _sacc_check["202"]
        check["sacc20"] += _sacc_check["20"]
        check["sacc02"] += _sacc_check["02"]

        ###PSO sanity check
        check["pso"] += len(_evt.query("evt==3"))

        # pp: change short PSOs to fixations; not used
        #        thres_pd = kwargs['thres_pd']/1000.
        #        thres_pd_s = round_up(thres_pd*fs)
        #        _pdur = _evt.query('evt==3 and (dur<@thres_pd or dur_s<@thres_pd_s)')
        #        check['short_pso']+=len(_pdur)
        #        self.check_inds['short_pso'] = _pdur.index.values
        #        if pp:
        #            _evt.loc[_pdur.index, 'evt'] = 1

        # pp: remove unreasonable psos
        _pso_check = {"13": 0, "03": 0, "23": 0}
        seq = "".join(map(str, _evt["evt"]))
        for pattern in list(_pso_check.keys()):
            _check = np.array([m.start() for m in re.finditer("(?=%s)" % pattern, seq)])
            if not (len(_check)):
                continue

            _pso_check[pattern] += len(_check)
            self.check_inds["pso%s" % pattern] = _check
            # pp: change PSOs after fixations to fixations
            if pp:
                if pattern == "13":
                    assert (_evt.loc[_check + 1, "evt"] == 3).all()
                    _evt.loc[_check + 1, "evt"] = 1
                # pp: change PSOs after undef to undef
                if pattern == "03":
                    assert (_evt.loc[_check + 1, "evt"] == 3).all()
                    _evt.loc[_check + 1, "evt"] = 0
        check["pso23"] += _pso_check["23"]
        check["pso13"] += _pso_check["13"]
        check["pso03"] += _pso_check["03"]

        ###fixation sanity check
        # unfold and recalculate event data
        _evt_unfold = [
            _e
            for _, e in _evt.iterrows()
            for _e in itertools.repeat(e["evt"], int(np.diff(e[["s", "e"]])))
        ]
        _etdata = copy.deepcopy(etdata)
        _etdata.data["evt"] = _evt_unfold
        _etdata.calc_evt(fast=True)
        _evt = _etdata.evt

        check["fixations"] += len(_evt.query("evt==1"))

        # pp: remove short fixations
        thres_fd = kwargs["thres_fd"] / 1000.0
        thres_fd_s = round_up(thres_fd * fs)
        _fdur = _evt.query("evt==1 and (dur<@thres_fd or dur_s<@thres_fd_s)")
        check["short_fixations"] += len(_fdur)
        self.check_inds["short_fixations"] = _fdur.index.values

        # TODO:
        # check fixation merge
        if pp:
            _inds = np.array(_fdur.index)
            _evt.loc[_inds, "evt"] = 0
        #            #check if there are saccades or psos left around newly taged undef
        #            #so basically +- 2 events around small fixation
        #            _inds = np.unique(np.concatenate((_inds, _inds+1, _inds-1, _inds+2, _inds-2)))
        #            _inds = _inds[(_inds>-1) & (_inds<len(_evt))]
        #            _mask =_evt.loc[_inds, 'evt'].isin([2, 3])
        #            _evt.loc[_inds[_mask.values], 'evt'] = 0

        ##return result
        _evt_unfold = [
            _e
            for _, e in _evt.iterrows()
            for _e in itertools.repeat(e["evt"], int(np.diff(e[["s", "e"]])))
        ]
        assert len(_evt_unfold) == len(status)

        status[np.array(_evt_unfold) == 0] = False

        return np.array(_evt_unfold), status, check, self.check_inds


def post_process_csv(input_path, output_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_path)

    # Perform post processing on the loaded data
    # Add your post processing logic here

    # Save the processed data to a new CSV file
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Example usage
    input_file = "/path/to/input.csv"
    output_file = "/path/to/output.csv"
    post_process_csv(input_file, output_file)


def fixation_merge(df, threshold=1):
    # Get the indices where label changes
    label_changes = (df["evt"] != df["evt"].shift()).cumsum()

    # Group the DataFrame by label changes
    grouped = df.groupby(label_changes)

    # Calculate the pythagorean distances between consecutive groupings with label 1, and merge them if the distance is below the threshold
    distances = []
    merge_counter=0
    for group_label, group_data in grouped:
        if 1 in group_data["evt"].values:
            group_indices = group_data.index
            if len(group_indices) > 1 and group_label != label_changes.max():
                last_sample_label1 = group_data[group_data["evt"] == 1].iloc[-1]
                next_sample_label1 = None
                for i in range(group_label + 1, label_changes.max() + 1):
                    if 1 in grouped.get_group(i)["evt"].values:
                        next_sample_label1 = (
                            grouped.get_group(i).loc[grouped.get_group(i)["evt"] == 1].iloc[0]
                        )
                        break
                    else:
                        next_sample_label1 = df.iloc[-1] # If no more label 1 samples, use the last sample
                distance = (
                    (last_sample_label1["x"] - next_sample_label1["x"]) ** 2
                    + (last_sample_label1["y"] - next_sample_label1["y"]) ** 2
                ) ** 0.5
                if distance < threshold:
                    indices_to_update = range(
                        last_sample_label1.name + 1, next_sample_label1.name
                    )
                    df.loc[indices_to_update, "evt"] = 1
                    merge_counter+=1
                distances.append(distance)
    #print("Pythagorean distances between consecutive groupings with label 1:")
    #print(distances)
    print(f"Merged {merge_counter} fixations due to them being closer than {threshold} degrees.")
    return df


def post_process(df, config):
    # Perform post processing on the loaded data
    index = df.index
    pp = hpp()
    etdata = ETData(df)
    # hard post-processing
    etdata.data["evt"], etdata.data["status"], pp_rez, pp_inds = pp.run_pp(
        etdata, **config["pp_kwargs"]
    )
    et_df = pd.DataFrame(etdata.data)

    et_df = fixation_merge(et_df, config["sacc_minimum_distance"])
    et_df.set_index(index, inplace=True)
    
    print(pp_rez)
    print(pp_inds)
    return et_df
