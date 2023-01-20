import numpy as np
import pandas as pd
import torch

class ManyHotEncoder:
    """"
        Adapted after DecisionEncoder.find_contiguous_regions method in
        https://github.com/DCASE-REPO/dcase_util/blob/master/dcase_util/data/decisions.py

        Encode labels into numpy arrays where 1 correspond to presence of the class and 0 absence.
        Multiple 1 can appear on the same line, it is for multi label problem.
    Args:
        labels: list, the classes which will be encoded
        n_frames: int, (Default value = None) only useful for strong labels. The number of frames of a segment.
    Attributes:
        labels: list, the classes which will be encoded
        n_frames: int, only useful for strong labels. The number of frames of a segment.
    """

    def __init__(
        self, taxonomy, use_taxo_fine, audio_len, frame_len, frame_hop, net_pooling=1, fs=32000
    ):

        
        self.taxonomy_coarse = taxonomy["coarse"]
        self.taxonomy_fine = taxonomy["fine"]
        
        if use_taxo_fine:
            self.taxonomy = self.taxonomy_fine
            labels = self.taxonomy["class_labels"]
        else:
            self.taxonomy = self.taxonomy_coarse
            labels = self.taxonomy["class_labels"]
        
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()

        self.labels = labels
        self.audio_len = audio_len
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.fs = fs
        self.net_pooling = net_pooling
        n_frames = self.audio_len * self.fs
        # self.n_frames = int(
        #     int(((n_frames - self.frame_len) / self.frame_hop)) / self.net_pooling
        # )
        self.n_frames = int(int((n_frames / self.frame_hop)) / self.net_pooling)
        self.ftc_matrix = torch.tensor(self.compute_fine_to_coarse_matrix())#, device="cuda:0")


    def compute_fine_to_coarse_matrix(self):
        classes_fine = self.taxonomy_fine["class_labels"]
        classes_coarse = self.taxonomy_coarse["class_labels"]
        t_matrix = np.zeros((len(classes_fine), len(classes_coarse)))

        for k in self.taxonomy_fine["SONYC"].keys():
            c_fine = self.taxonomy_fine["SONYC"][k]
            if c_fine == "no-annotation":
                continue
            c_coarse = self.taxonomy_coarse["SONYC"][k]
            idx_fine = classes_fine.index(c_fine)
            idx_coarse = classes_coarse.index(c_coarse)
            
            t_matrix[idx_fine, idx_coarse] = 1

        for k in self.taxonomy_fine["SINGA-PURA"].keys():
            c_fine = self.taxonomy_fine["SINGA-PURA"][k]
            c_coarse = self.taxonomy_coarse["SINGA-PURA"][k]
            if c_fine == "no-annotation":
                continue
            idx_fine = classes_fine.index(c_fine)
            idx_coarse = classes_coarse.index(c_coarse)
            
            t_matrix[idx_fine, idx_coarse] = 1
        
        return t_matrix

    def fine_to_coarse(self, prob):
        """Transform a fine-grained prediction or groundtruth into a coarse-grained one

        Args:
            prob (tensor): fine-grained prediction or groundtruth tensor

        Returns:
            (tensor): coarse-grained prediction or groundtruth tensor

        """

        if prob.dim() == 3:
            return torch.max(prob.transpose(1,2)[:,:,:,None]*self.ftc_matrix.to(prob),axis=-2,keepdims=True).values.squeeze(-2).transpose(1,2)
        elif prob.dim() == 2:
            return torch.max(prob[:,:,None]*self.ftc_matrix.to(prob),axis=-2,keepdims=True).values.squeeze(-2)
        

    def convert_label_fine_to_coarse(self, label):
        idx_fine = self.taxonomy_fine["class_labels"].index(label)
        return self.taxonomy_coarse["class_labels"][(np.where(self.ftc_matrix[idx_fine])[0][0])]
    
    def encode_weak(self, labels, dset):
        """ Encode a list of weak labels into a numpy array

        Args:
            labels: list, list of labels to encode (to a vector of 0 and 1)

        Returns:
            numpy.array
            A vector containing 1 for each label, and 0 everywhere else
        """
        
        # Convert into the unified taxonomy label
        labels = list(map(lambda label: self.taxonomy[dset][label], labels))
        y = np.zeros(len(self.labels))
        for label in labels:
            if not pd.isna(label) and label != "no-annotation":
                i = self.labels.index(label)
                y[i] = 1
        return y

    def _time_to_frame(self, time):
        samples = time * self.fs
        frame = (samples) / self.frame_hop
        return np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames)

    def _frame_to_time(self, frame):
        frame = frame * self.net_pooling / (self.fs / self.frame_hop)
        return np.clip(frame, a_min=0, a_max=self.audio_len)

    def encode_strong_df(self, label_df, dset):
        """Encode a list (or pandas Dataframe or Serie) of strong labels, they correspond to a given filename

        Args:
            label_df: pandas DataFrame or Series, contains filename, onset (in frames) and offset (in frames)
                If only filename (no onset offset) is specified, it will return the event on all the frames
                onset and offset should be in frames
        Returns:
            numpy.array
            Encoded labels, 1 where the label is present, 0 otherwise
        """

        assert any(
            [x is not None for x in [self.audio_len, self.frame_len, self.frame_hop]]
        )

        samples_len = self.n_frames
        #if type(label_df) is str:
        #    if label_df == "empty":
        #        y = np.zeros((samples_len, len(self.labels))) - 1
        #        return y
        y = np.zeros((samples_len, len(self.labels)))
        if type(label_df) is pd.DataFrame:
            if {"onset", "offset", "event_label"}.issubset(label_df.columns):
                for _, row in label_df.iterrows():
                    unified_label = self.taxonomy[dset][row["event_label"]]
                    if not pd.isna(row["event_label"]) and unified_label != "no-annotation":
                        i = self.labels.index(unified_label)
                        onset = int(self._time_to_frame(row["onset"]))
                        offset = int(np.ceil(self._time_to_frame(row["offset"])))
                        y[
                            onset:offset, i
                        ] = 1  # means offset not included (hypothesis of overlapping frames, so ok)

        else:
            raise NotImplementedError(
                "To encode_strong, type is pandas.Dataframe with onset, offset and event_label"
                "columns,type given: {}".format(type(label_df))
            )
        return y

    def decode_weak(self, labels):
        """ Decode the encoded weak labels
        Args:
            labels: numpy.array, the encoded labels to be decoded

        Returns:
            list
            Decoded labels, list of string

        """
        result_labels = []
        for i, value in enumerate(labels):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def decode_strong(self, labels, taxo_level=None):
        """ Decode the encoded strong labels
        Args:
            labels: numpy.array, the encoded labels to be decoded
        Returns:
            list
            Decoded labels, list of list: [[label, onset offset], ...]

        """
        if len(labels) == len(self.taxonomy_coarse["class_labels"]):
            class_labels = self.taxonomy_coarse["class_labels"]
        if len(labels) == len(self.taxonomy_fine["class_labels"]):
            class_labels = self.taxonomy_fine["class_labels"]
        elif taxo_level == "coarse":
            class_labels = self.taxonomy_coarse["class_labels"]
        elif taxo_level == "fine":
            class_labels = self.taxonomy_fine["class_labels"]
        result_labels = []
        for i, label_column in enumerate(labels):
            change_indices = self.find_contiguous_regions(label_column)

            # append [label, onset, offset] in the result list
            for row in change_indices:
                result_labels.append(
                    [
                        class_labels[i],
                        self._frame_to_time(row[0]),
                        self._frame_to_time(row[1]),
                    ]
                )
        return result_labels

    def find_contiguous_regions(self, activity_array):
        """Find contiguous regions from bool valued numpy.array.
        Transforms boolean values for each frame into pairs of onsets and offsets.

        Parameters
        ----------
        activity_array : numpy.array [shape=(t)]
            Event activity array, bool values

        Returns
        -------
        numpy.ndarray [shape=(2, number of found changes)]
            Onset and offset indices pairs in matrix

        """

        # Find the changes in the activity_array
        change_indices = np.logical_xor(activity_array[1:], activity_array[:-1]).nonzero()[0]

        # Shift change_index with one, focus on frame after the change.
        change_indices += 1

        if activity_array[0]:
            # If the first element of activity_array is True add 0 at the beginning
            change_indices = np.r_[0, change_indices]

        if activity_array[-1]:
            # If the last element of activity_array is True, add the length of the array
            change_indices = np.r_[change_indices, activity_array.size]

        # Reshape the result into two columns
        return change_indices.reshape((-1, 2))

    
    def state_dict(self):
        return {
            "labels": self.labels,
            "audio_len": self.audio_len,
            "frame_len": self.frame_len,
            "frame_hop": self.frame_hop,
            "net_pooling": self.net_pooling,
            "fs": self.fs,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        labels = state_dict["labels"]
        audio_len = state_dict["audio_len"]
        frame_len = state_dict["frame_len"]
        frame_hop = state_dict["frame_hop"]
        net_pooling = state_dict["net_pooling"]
        fs = state_dict["fs"]
        return cls(labels, audio_len, frame_len, frame_hop, net_pooling, fs)