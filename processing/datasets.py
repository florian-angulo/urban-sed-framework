from tkinter import N
import pandas as pd 
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from loguru import logger

class ConcatDatasetUrban(torch.utils.data.ConcatDataset):
    # Expanding the ConcatDataset class with a collate function
    def __init__(self, datasets, encoder, batch_sizes, n_samples=320000, analyse_proximity=False):
        super().__init__(datasets)
        self.batch_sizes = batch_sizes
        self.n_samples = n_samples

        self.taxo_name = encoder.taxonomy["name"]
        self.n_classes = len(encoder.taxonomy["class_labels"])
        self.n_frames = encoder.n_frames
        self.n_samples = encoder.audio_len * encoder.fs
        hdf5_path = None
        self.analyse_proximity = analyse_proximity
        
        for dset in self.datasets:
            hdf5_path = hdf5_path if hdf5_path is not None else dset.hdf5_path
            if hdf5_path != dset.hdf5_path:
                raise ValueError(f"Datasets must use the same hdf5_path, one mismatch found : {hdf5_path} != {dset.hdf5_path}")
            if self.taxo_name != dset.taxo_name:
                raise ValueError(f"Datasets must use the same taxonomy, one mismatch found : {self.taxo_name} != {dset.taxo_name}")
            if self.n_classes != dset.n_classes:
                raise ValueError(f"Unexpected error : mismatch in the number of classes from the common taxonomy : {self.n_classes} != {dset.n_classes}")

            # Commented for now because filenames are always returned
            ''' 
            if len(self.batch_sizes) > 1 and dset.return_filename:
                raise ValueError("Multiple batch_sizes specified in val or test mode")
            
            if len(self.batch_sizes) == 1 and  not dset.return_filename:
                raise ValueError("batch_sizes should have more than one element in training mode, got len batch_sizes = {}")
            '''
        
        self.hdf5_path = hdf5_path

        
    def collate_fn(self, batch):
        if not hasattr(self, "hdf5"):
            self.hdf5 = h5py.File(self.hdf5_path, "r")

        indices, filenames = zip(*batch)
        indices = np.array(indices).astype(int)
        filenames = np.array(filenames).astype(str)        
        
        if "test" in self.hdf5_path:
            group_name_gt_SGP = "groundtruth_with_proximity" if self.analyse_proximity else "groundtruth"
            group_name_gt_SONYC = "groundtruth_with_proximity_perso" if self.analyse_proximity else "groundtruth"
        else:
            group_name_gt_SGP = "groundtruth"
            group_name_gt_SONYC = "groundtruth"

        
        if len(self.batch_sizes) > 1:
            indx_strong, indx_weak, indx_unlabelled = self.batch_sizes
            
            audio = np.zeros((indx_strong + indx_weak + indx_unlabelled, self.n_samples))
            labels = np.zeros((indx_strong + indx_weak + indx_unlabelled, self.n_classes, self.n_frames))

            if indx_strong > 0:
                sort_order_strong = np.argsort(indices[:indx_strong])
                audio[:indx_strong] = self.hdf5["SINGA-PURA"]["audio_32k"][np.sort(indices[:indx_strong])]
                labels[:indx_strong] = self.hdf5["SINGA-PURA"]["groundtruth"][self.taxo_name][np.sort(indices[:indx_strong])]
                filenames[:indx_strong] = filenames[:indx_strong][sort_order_strong.astype(int)]
            if indx_weak > 0 :  
                sort_order_weak = np.argsort(indices[indx_strong:indx_strong+indx_weak])
                audio[indx_strong:indx_strong+indx_weak] = self.hdf5["SONYC"]["audio_32k"][np.sort(indices[indx_strong:indx_strong+indx_weak])]
                labels[indx_strong:indx_strong+indx_weak, :, 0] = self.hdf5["SONYC"]["groundtruth"][self.taxo_name][np.sort(indices[indx_strong:indx_strong+indx_weak])]            
                filenames[indx_strong:indx_strong+indx_weak] = filenames[indx_strong:indx_strong+indx_weak][sort_order_weak.astype(int)]
            if indx_unlabelled > 0:
                sort_order_unlab = np.argsort(indices[-indx_unlabelled:])
                audio[-indx_unlabelled:] = self.hdf5["unlabelled_SINGA-PURA"]["audio_32k"][np.sort(indices[-indx_unlabelled:])]
                filenames[-indx_unlabelled:] = filenames[-indx_unlabelled:][sort_order_unlab.astype(int)]                    
        else:
            from_SONYC = [s[0] != '[' for s in filenames]
            
            indices_SONYC = indices[from_SONYC]
            indices_SINGAPURA = indices[~np.array(from_SONYC)]
            
            sort_order_SONYC = np.argsort(indices_SONYC)
            sort_order_SINGAPURA = np.argsort(indices_SINGAPURA)
            
            n_SONYC = len(indices_SONYC)
            n_SINGAPURA = len(indices_SINGAPURA)
                        
            audio = np.zeros((n_SONYC + n_SINGAPURA, self.n_samples))
            labels = np.zeros((n_SONYC + n_SINGAPURA, self.n_classes, self.n_frames))
            mem_filenames = np.copy(filenames)
            
            if n_SINGAPURA > 0:
                audio[:n_SINGAPURA] = self.hdf5["SINGA-PURA"]["audio_32k"][indices_SINGAPURA[sort_order_SINGAPURA.astype(int)]]
                labels[:n_SINGAPURA] = self.hdf5["SINGA-PURA"][group_name_gt_SGP][self.taxo_name][indices_SINGAPURA[sort_order_SINGAPURA.astype(int)]]
                filenames[:n_SINGAPURA] = mem_filenames[~np.array(from_SONYC)][sort_order_SINGAPURA.astype(int)]
                
            if n_SONYC > 0:
                audio[n_SINGAPURA:] = self.hdf5["SONYC"]["audio_32k"][indices_SONYC[sort_order_SONYC.astype(int)]]
                labels[n_SINGAPURA:, :, 0] = self.hdf5["SONYC"][group_name_gt_SONYC][self.taxo_name][indices_SONYC[sort_order_SONYC.astype(int)]]    
                filenames[n_SINGAPURA:] = mem_filenames[from_SONYC][sort_order_SONYC.astype(int)]  

        
        # Centering audio
        audio -= np.mean(audio, axis=(1,), keepdims=True)

        return torch.from_numpy(audio).float(), torch.from_numpy(labels).float(), filenames


class HDF5_dataset(Dataset):
    def __init__(
        self,
        hdf5_path,
        dname,
        encoder,
        remove_non_target=True,
    ):
        self.hdf5_path = hdf5_path
        self.taxo_name = encoder.taxonomy["name"]
        self.n_classes = len(encoder.taxonomy["class_labels"])
        with h5py.File(hdf5_path, 'r') as hf:
            filenames = hf[dname]["filenames"]
            if (dname != "unlabelled_SINGA-PURA") and remove_non_target:
                labels = hf[dname]["groundtruth"][encoder.taxonomy["name"]]
                axis = (1,2) if dname == "SINGA-PURA" else 1
                self.ids = np.where(np.any(labels, axis=axis))[0]
                self.filenames = filenames[self.ids].astype(str)                
            else:
                self.ids = np.arange(len(filenames))
                self.filenames = np.array(filenames).astype(str)
            
            if dname == "SINGA-PURA":
                self.durations, self.groundtruths = self._generate_eval_dfs(hf, encoder)
            else:
                self.durations = None
                self.groundtruths = None
                
            

        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        return self.ids[item], self.filenames[item]

    def _generate_eval_dfs(self, hf, encoder):
        # generate duration dataframe
        durations = pd.DataFrame.from_dict({"filename": self.filenames, "duration": [10.0] * len(self.ids)})
        # generate groundtruth dataframe with unified taxonomy
        groundtruths = pd.DataFrame(columns=["filename", "onset", "offset", "event_label"])
        filenames = []
        filenames_monoph = []
        filenames_lowpolyph = []
        filenames_highpolyph = []
        event_labels = []
        onsets = []
        offsets = []
        for i, fname in enumerate(self.filenames):
            labels = hf["SINGA-PURA"]["groundtruth"][encoder.taxonomy_coarse["name"]][self.ids[i]]
            polyphony = np.max(np.sum(labels, axis=0))
            if polyphony == 1:
                filenames_monoph.append(fname)
            if polyphony == 2:
                filenames_lowpolyph.append(fname)
            if polyphony > 2:
                filenames_highpolyph.append(fname)
            
            
            decoded_labels = encoder.decode_strong(labels, taxo_level="coarse")
            for result_label in decoded_labels:
                filenames.append(fname)
                event_labels.append(result_label[0])
                onsets.append(result_label[1])
                offsets.append(result_label[2])
            
        groundtruths = pd.DataFrame.from_dict({"filename": filenames, "onset": onsets, "offset": offsets, "event_label": event_labels})
        
        self.groundtruths_monoph = groundtruths[groundtruths["filename"].isin(filenames_monoph)]
        self.groundtruths_lowpolyph = groundtruths[groundtruths["filename"].isin(filenames_lowpolyph)]
        self.groundtruths_highpolyph = groundtruths[groundtruths["filename"].isin(filenames_highpolyph)]
        self.durations_monoph = pd.DataFrame.from_dict({"filename": filenames_monoph, "duration": [10.0] * len(filenames_monoph)})
        self.durations_lowpolyph = pd.DataFrame.from_dict({"filename": filenames_lowpolyph, "duration": [10.0] * len(filenames_lowpolyph)})
        self.durations_highpolyph = pd.DataFrame.from_dict({"filename": filenames_highpolyph, "duration": [10.0] * len(filenames_highpolyph)})
        
        # generate groundtruth dataframe with unified taxonomy for near and moving events of the test set
        if "test" in self.hdf5_path:
            filenames_near = []
            filenames_far = []
            event_labels_near = []
            event_labels_far = []
            onsets_near = []
            onsets_far = []
            offsets_near = []
            offsets_far = []
        
            for i, fname in enumerate(self.filenames):
                
                # near events
                labels = hf["SINGA-PURA"]["groundtruth_with_proximity"][encoder.taxonomy_coarse["name"]][self.ids[i]]
                
                labels_near = np.logical_or(labels == 2, labels == 5)
                labels_far = (labels == 3)
                
                decoded_labels = encoder.decode_strong(labels_near, taxo_level="coarse")
                for result_label in decoded_labels:
                    filenames_near.append(fname)
                    event_labels_near.append(result_label[0])
                    onsets_near.append(result_label[1])
                    offsets_near.append(result_label[2])
                
                #far events
                decoded_labels = encoder.decode_strong(labels_far, taxo_level="coarse")
                for result_label in decoded_labels:            
                    filenames_far.append(fname)
                    event_labels_far.append(result_label[0])
                    onsets_far.append(result_label[1])
                    offsets_far.append(result_label[2])
            
            self.groundtruths_near = pd.DataFrame.from_dict({"filename": filenames_near, "onset": onsets_near, "offset": offsets_near, "event_label": event_labels_near})
            self.durations_near = pd.DataFrame.from_dict({"filename": filenames_near, "duration": [10.0] * len(filenames_near)})
            self.groundtruths_far = pd.DataFrame.from_dict({"filename": filenames_far, "onset": onsets_far, "offset": offsets_far, "event_label": event_labels_far})
            self.durations_far = pd.DataFrame.from_dict({"filename": filenames_far, "duration": [10.0] * len(filenames_far)})
    
    
        return durations, groundtruths