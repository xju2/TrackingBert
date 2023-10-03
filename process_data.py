"""Read TrackML data.
https://www.kaggle.com/c/trackml-particle-identification/overview
Assume the files are like:
detectors.csv
trackml/event000001000-hits.csv
trackml/event000001000-cells.csv
trackml/event000001000-particles.csv
trackml/event000001000-truth.csv
"""
import os
import glob
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from acctrack.io.base import BaseReader
from acctrack.io import MeasurementData
from acctrack.io.utils import make_true_edges
from acctrack.io.trackml_cell_info import add_cluster_shape
from acctrack.io.trackml_detector import load_detector


__all__ = ['TrackMLReader', 'select_barrel_hits', 'remove_noise_hits']

def select_barrel_hits(hits):
    """Select barrel hits.
    """
    vlids = [(8,2), (8,4), (8,6), (8,8), (13,2), (13,4), (13,6), (13,8), (17,2), (17,4)]
    n_det_layers = len(vlids)
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])
    return hits

def remove_noise_hits(hits):
    """Remove noise hits.
    """
    # Remove noise hits
    hits = hits[hits.hit_type != 'noise']
    return hits

class TrackMLReader(BaseReader):
    """
    TrackML Reader copied from the acctrack library, with adjustment on data processing.
    """
    def __init__(self, basedir, detector_path, name="TrackMLReader") -> None:
        super().__init__(basedir, name)

        # count how many events in the directory
        all_evts = glob.glob(os.path.join(
            self.basedir, "event*-hits.csv"))

        self.nevts = len(all_evts)
        pattern = "event([0-9]*)-hits.csv"
        self.all_evtids = sorted([
            int(re.search(pattern, os.path.basename(x)).group(1).strip())
                for x in all_evts])

        print("total {} events in directory: {}".format(
            self.nevts, self.basedir))
        
        #detector_path = os.path.join(self.basedir, "../detectors.csv")
        _, self.detector = load_detector(detector_path)


    def read(self, evtid: int = None) -> MeasurementData:
        """Read one event from the input directory"""

        if (evtid is None or evtid < 1) and self.nevts > 0:
            evtid = self.all_evtids[0]
            print("read event {}".format(evtid))

        prefix = os.path.join(self.basedir, "event{:09d}".format(evtid))
        hit_fname = "{}-hits.csv".format(prefix)
        cell_fname = "{}-cells.csv".format(prefix)
        particle_fname = "{}-particles.csv".format(prefix)
        truth_fname = "{}-truth.csv".format(prefix)

        # read all files
        hits = pd.read_csv(hit_fname)
        r = np.sqrt(hits.x**2 + hits.y**2)
        phi = np.arctan2(hits.y, hits.x)
        hits = hits.assign(r=r, phi=phi)

        # read truth info about hits and particles
        truth = pd.read_csv(truth_fname)
        particles = pd.read_csv(particle_fname)
        ### add dummy particle information for noise hits
        ### whose particle ID is zero.
        ### particle_id,vx,vy,vz,px,py,pz,q,nhits
        particles.loc[len(particles.index)] = [0, 0, 0, 0, 0.00001, 0.00001, 0.00001, 0, 0]
        truth.merge(particles, on='particle_id', how='left')
        truth = truth.assign(pt=np.sqrt(truth.tpx**2 + truth.tpy**2))


        #hits = hits.merge(truth['hit_id', 'particle_id',
                                 #'vx', 'vy', 'vz', 
                                 #'pt', 'weight'] + particles['vx', 'vy', 'vz'], on='hit_id')

        #true_edges = make_true_edges(hits)
        cells = pd.read_csv(cell_fname)
        hits = add_cluster_shape(hits, cells, self.detector)

        #hits = hits.assign(R=np.sqrt( (hits.x - hits.vx)**2 + (hits.y - hits.vy)**2 + (hits.z - hits.vz)**2 ))
        #hits = hits.sort_values('R').reset_index(drop=True).reset_index(drop=False)
        
        """
        data = MeasurementData(
            hits=None,
            measurements=None,
            meas2hits=None,
            spacepoints=hits,
            particles=particles,
            #true_edges=true_edges,
            event_file=os.path.abspath(prefix),
        )
        """
        return hits, particles, truth


def get_tracks(df):
    """
    Get the tracks given a dataframe.
    """
    particlegroup = df.groupby('particle_id')
    pids = df.particle_id.unique()
    
    particles = [particlegroup.get_group(pid) for pid in pids if pid != 0]
    return particles

def process(i_file, n_evt=100, r_cutoff=200, nhits_cutoff=(3,8), save_prefix="."):
    """
    Process n events into a file indexed i.
    
    Parameters
    ----------
    i_file : int
        The index of the file to be saved into.
    n_evt : int
        Number of events to be processed and saved into this file.
    r_cutoff : float
        The cutoff applied to r.
    nhits_cutoff : (int, int)
        The cutoff applied to the number of hits per track, in the format
        of (min_num_hits, max_num_hits).
    save_prefix : str
        The prefix of the file path to save to.
    """
    
    init_evt = i_file * n_evt
    final_evt = (i_file + 1) * n_evt

    all_radius = []
    all_Pt = []
    def pair_hits(track, hits):
        all_umid_tuples = []
        for hit_id in track['hit_id']:
            row = hits.loc[hits['hit_id']==hit_id]
            tru = truth.loc[hits['hit_id']==hit_id]
            if all(row.r <= r_cutoff):
                all_umid_tuples += [(int(row.volume_id), int(row.layer_id), int(row.module_id))]
                all_radius.append(row.r)
                all_Pt.append(tru.pt)
        return all_umid_tuples

    all_evt_seq = []
    all_seq_len = []
    idx = 0
    for evt_id in tqdm(reader.all_evtids, desc=f'File {i_file}'):
        if idx < init_evt:
            idx += 1
            continue
        if idx >= final_evt:
            break
        hits, particles, truth = reader.read(evt_id)

        all_tracks = get_tracks(truth)
        seq_len = []
        for i in range(len(all_tracks)):
            track = all_tracks[i]
            all_pairs = pair_hits(track, hits)
            n_hits = len(all_pairs)
            if n_hits <= nhits_cutoff[0] or n_hits > nhits_cutoff[1]: # apply the cut on sequence length
                continue
            seq_len += [n_hits]
            event_seq = np.array([(umid_dict[tuple(pair)]) for pair in all_pairs], dtype=np.int32)
            all_evt_seq.append(event_seq)
        all_seq_len.append(seq_len)
        idx += 1
        
    np.savez(f'{save_prefix}/train_{i_file}.npz', seq=all_evt_seq, umid_dict=umid_dict, length=all_seq_len, Pt=all_Pt, R=all_radius)
    print(f">>> Finished file {i_file}")

if __name__ == '__main__':
    
    ### Parameters to Change ###
    input_dir = "/global/cfs/cdirs/m3443/data/trackml-kaggle/train_all"
    detector_path = 'detectors.csv' # the path for this detectors.csv file
    data_saving_prefix = 'train_data' # the directory to save output data files
    r_cutoff = 200 # the cutoff radius
    nhits_cutoff = (3, 8) # the cutoff range for the number of hits per track
    n_evt_per_file = 100 # number of events to save per file
    num_works = 10 # number of runs in parallel
    
    
    reader = TrackMLReader(input_dir, detector_path=detector_path)
    detector = pd.read_csv(detector_path)

    detector_umid = np.stack([detector.volume_id, detector.layer_id, detector.module_id], axis=1)
    umid_dict = {}
    index = 1
    for i in detector_umid:
        umid_dict[tuple(i)] = index
        index += 1

    n_files = reader.nevts // n_evt_per_file + 1
    with Pool(num_works) as p:
        run_func = partial(process, n_evt=n_evt_per_file, 
                           r_cutoff=r_cutoff, nhits_cutoff=nhits_cutoff, 
                           save_prefix=data_saving_prefix) 
        p.map(run_func, list(range(n_files)))
    
    print(">>> Completed processing all events")
    