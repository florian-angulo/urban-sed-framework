import pandas as pd 
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import soundata
import h5py


#* Not useful with hdf5 dataloaders
def to_mono(mixture, random_ch=False):

    if mixture.ndim > 1:  # multi channel
        if not random_ch:
            mixture = torch.mean(mixture, 0)
        else:  # randomly select one channel
            indx = np.random.randint(0, mixture.shape[0])
            mixture = mixture[indx]
    return mixture


def pad_audio(audio, target_len):
    if audio.shape[-1] < target_len:
        audio = torch.nn.functional.pad(
            audio, (0, target_len - audio.shape[-1]), mode="constant"
        )
        padded_indx = [target_len / len(audio)]
    else:
        padded_indx = [1.0]

    return audio, padded_indx


def read_audio(file, multisrc, random_channel, pad_to):
    mixture, fs = torchaudio.load(file)
    if not multisrc:
        mixture = to_mono(mixture, random_channel)

    if pad_to is not None:
        mixture, padded_indx = pad_audio(mixture, pad_to)
    else:
        padded_indx = [1.0]

    mixture = mixture.float()
    return mixture, padded_indx

class HDF5_SINGAPURA_labelled(Dataset):
    def __init__(
        self,
        hdf5_path,
        csv_path,
        encoder,
        return_filename=False,
        taxonomy=None,
    ):
        
        
        dset = soundata.initialize("singapura", "/tsi/dcase/SINGA-PURA/")
        self.clips = dset.load_clips()
        df = pd.read_csv(csv_path)
        self.ids = df["clip_id"].to_list()
        self.encoder = encoder
        self.return_filename = return_filename
        
        self.hdf5_path = hdf5_path
        
        self.events = {}
        for filename in self.ids:
            onset = self.clips[filename].events.annotations[0].intervals[:,0]
            offset = self.clips[filename].events.annotations[0].intervals[:,1]
            label = self.clips[filename].events.annotations[0].labels
            self.events[filename] = pd.DataFrame({"event_label": label, "onset":onset, "offset":offset})

        if taxonomy is not None:
            durations, groundtruths = self._generate_eval_dfs(taxonomy)
        else:
            durations = None
            groundtruths = None

        self.durations = durations
        self.groundtruths = groundtruths
            
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if not hasattr(self, "hdf5_audio"):
            self.hdf5_audio = h5py.File(self.hdf5_path, "r")
        mixture = self.hdf5_audio["SINGA-PURA/audio_32k"][item]
        filename = self.ids[item]
        labels = self.events[filename]
        # to steps
        strong = self.encoder.encode_strong_df(pd.DataFrame(labels), "SINGA-PURA")
        strong = torch.from_numpy(strong).float()

        if self.return_filename:
            return mixture, strong.transpose(0, 1), self.ids[item]
        else:
            return mixture, strong.transpose(0, 1)


    def _generate_eval_dfs(self, taxonomy):
        # generate duration dataframe
        durations = pd.DataFrame.from_dict({"filename": self.ids, "duration": [10.0] * len(self.ids)})
        # generate groundtruth dataframe with unified taxonomy
        groundtruths = pd.DataFrame(columns=["filename", "onset", "offset", "event_label"])
        for fname in self.ids:
            labels = self.events[fname]
            strong = self.encoder.encode_strong_df(pd.DataFrame(labels), "SINGA-PURA")
            decoded_labels = self.encoder.decode_strong(strong.transpose(0, 1))
            for result_label in decoded_labels:
                event_label = result_label[0]
                onset = result_label[1]
                offset = result_label[2]
                new_row = {"filename": fname, "onset": onset, "offset": offset, "event_label": event_label}
                #append to the dataframe
                groundtruths = groundtruths.append(new_row, ignore_index=True)
                        
        return durations, groundtruths

class HDF5_SONYC_Dataset(Dataset):
    def __init__(
        self,
        hdf5_path,
        csv_path,
        encoder,
        return_filename=False
    ):
        
        
        df = pd.read_csv(csv_path)
        self.ids = df["filename"].to_list()
        self.encoder = encoder
        self.return_filename = return_filename
        self.hdf5_path = hdf5_path
        
        
        self.events = {}
        for idx, filename in enumerate(self.ids):
            self.events[filename] = df["event_labels"][idx].split(";")

        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if not hasattr(self, "hdf5_audio"):
            self.hdf5_audio = h5py.File(self.hdf5_path, "r")
        mixture = self.hdf5_audio["SONYC/audio_32k"][item]
        filename = self.ids[item]
        # labels
        labels = self.events[filename]
        # check if labels exists:
        max_len_targets = self.encoder.n_frames
        weak = torch.zeros(max_len_targets, len(self.encoder.labels))
        if len(labels):
            weak_labels = self.encoder.encode_weak(labels, "SONYC")
            weak[0, :] = torch.from_numpy(weak_labels).float()

        out_args = [mixture, weak.transpose(0, 1)]

        if self.return_filename:
            out_args.append(filename)

        return out_args
  
        
    
class HDF5_SINGAPURA_unlabelled(Dataset):
    def __init__(
        self,
        hdf5_path,
        csv_path,
        encoder,
        return_filename=False
    ):
        
        
        df = pd.read_csv(csv_path)
        self.ids = df["filename"].to_list()
        self.encoder = encoder
        self.return_filename = return_filename
        self.hdf5_path = hdf5_path
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if not hasattr(self, "hdf5_audio"):
            self.hdf5_audio = h5py.File(self.hdf5_path, "r")
        mixture = self.hdf5_audio["unlabelled_SINGA-PURA/audio_32k"][item]
        filename = self.ids[item]

        max_len_targets = self.encoder.n_frames
        strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
        out_args = [mixture, strong.transpose(0, 1)]

        if self.return_filename:
            out_args.append(filename)

        return out_args