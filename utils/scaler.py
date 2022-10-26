import tqdm
import torch


class TorchScaler(torch.nn.Module):

    def __init__(self, batch_sizes, statistic="dataset", normtype="standard", dims=(2,), eps=1e-8):
        super(TorchScaler, self).__init__()
        assert statistic in ["dataset", "instance"]
        assert normtype in ["standard", "mean", "minmax"]
        if statistic == "dataset" and normtype == "minmax":
            raise NotImplementedError("dataset minmax normalization not implemented")
        self.statistic = statistic
        self.normtype = normtype
        self.dims = dims
        self.eps = eps
        self.batch_sizes = batch_sizes
        
    def load_state_dict(self, state_dict, stric=True):
        if self.statistic == "dataset":
            super(TorchScaler, self).load_state_dict(state_dict, stric)
    
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.statistic == "dataset":
            super(TorchScaler, self)._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs
            )
    
    def fit(self, dataloader, transform_func):
        """Scaler fitting

        Args:
            dataloader (DataLoader): training data Dataloader (only train ?)
            transform_func (lambda function, optional): Data transformation function. Defaults to lambdax:x[0].
        """
        idx = 0
        indx_strong, indx_weak, _ = self.batch_sizes
        for batch in tqdm.tqdm(dataloader):
            
            features = transform_func(batch)
            if idx == 0:
                
                batch_num = features.shape[0]
                # deriving masks for each dataset
                mask_strong = torch.zeros(batch_num, device=self.device).bool()
                mask_weak = torch.zeros(batch_num, device=self.device).bool()
                mask_unlab = torch.zeros(batch_num, device=self.device).bool()
                mask_strong[:indx_strong] = 1
                mask_weak[indx_strong : indx_weak + indx_strong] = 1
                mask_unlab[indx_strong + indx_weak:] = 1
                
                if torch.any(mask_strong):
                    mean_strong = torch.mean(features[mask_strong], self.dims, keepdim=True).mean(0).unsqueeze(0)
                    mean_squared_strong = (
                        torch.mean(features[mask_strong] ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                        )
                if torch.any(mask_weak):
                    mean_weak = torch.mean(features[mask_weak], self.dims, keepdim=True).mean(0).unsqueeze(0)
                    mean_squared_weak = (
                        torch.mean(features[mask_weak] ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                        )
                    
                if torch.any(mask_unlab):
                    mean_unlab = torch.mean(features[mask_unlab], self.dims, keepdim=True).mean(0).unsqueeze(0)
                    mean_squared_unlab = (
                        torch.mean(features[mask_unlab] ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                        )
            else:
                if torch.any(mask_strong):
                    mean_strong  += torch.mean(features[mask_strong], self.dims, keepdim=True).mean(0).unsqueeze(0)
                    mean_squared_strong += torch.mean(features[mask_strong] ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                if torch.any(mask_weak):
                    mean_weak  += torch.mean(features[mask_weak], self.dims, keepdim=True).mean(0).unsqueeze(0)
                    mean_squared_weak += torch.mean(features[mask_weak] ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                if torch.any(mask_unlab):
                    mean_unlab  += torch.mean(features[mask_unlab], self.dims, keepdim=True).mean(0).unsqueeze(0)
                    mean_squared_unlab += torch.mean(features[mask_unlab] ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                
            idx += 1
        
        if torch.any(mask_strong):
            mean_strong  /= idx
            mean_squared_strong /= idx
            self.register_buffer("mean_strong", mean_strong)
            self.register_buffer("mean_squared_strong", mean_squared_strong)

        if torch.any(mask_weak):
            mean_weak  /= idx
            mean_squared_weak /= idx
            self.register_buffer("mean_weak", mean_weak)
            self.register_buffer("mean_squared_weak", mean_squared_weak)
                    
        if torch.any(mask_unlab):
            mean_unlab  /= idx
            mean_squared_unlab /= idx
            self.register_buffer("mean_unlab", mean_unlab)
            self.register_buffer("mean_squared_unlab", mean_squared_unlab)
        
        
    def forward(self, tensor, mask_strong, mask_weak, mask_unlab):
        if torch.any(mask_strong):
            tensor[mask_strong] = self.normalize(tensor[mask_strong], "strong")
        if torch.any(mask_weak):
            tensor[mask_weak] = self.normalize(tensor[mask_weak], "weak")
        if torch.any(mask_unlab):
            tensor[mask_unlab] = self.normalize(tensor[mask_unlab], "unlab")
        
        return tensor
    
    def normalize(self, tensor, type_mask):
        if self.statistic == "dataset":
            if type_mask == "weak":
                mean = self.mean_weak
                mean_squared = self.mean_squared_weak
            elif type_mask == "strong":
                mean = self.mean_strong
                mean_squared = self.mean_squared_strong
            elif type_mask == "unlab":
                mean = self.mean_unlab
                mean_squared = self.mean_squared_unlab
            else:
                raise ValueError("wrong mask type")
            
            assert tensor.ndim == mean.ndim, "Pre-computed statistics"
            if self.normtype == "mean":
                return tensor - mean
            elif self.normtype == "standard":
                std = torch.sqrt(mean_squared - mean ** 2)
                return (tensor - mean) / (std + self.eps)
            else:
                raise NotImplementedError("normtype not implemented (scaler)")
        
        elif self.statistic == 'instance':
            
            mean = tensor.mean(dim=2, keepdims=True)
            std =  tensor.std(dim=2, keepdims=True)
            
            # we recompute the mean and std without outliers
            # (i.e. distant of more than one positive deviation from the mean)
            tensor_without_outliers = torch.where(torch.abs(tensor - mean) <= std, tensor, torch.nan)
            
            nanmean = torch.nanmean(tensor_without_outliers, self.dims, keepdim=True)
            nanmean_squared = torch.nanmean(torch.square(tensor_without_outliers), self.dims, keepdim=True)
            nanstd = torch.sqrt(nanmean_squared-torch.square(nanmean))
            
            if self.normtype == "mean":
                return tensor - nanmean
            elif self.normtype == "standard":
                return (tensor - nanmean) / (nanstd + self.eps)
            elif self.normtype == "minmax":
                return (tensor - torch.amin(tensor, dim=self.dims, keepdim=True)) / (
                    torch.amax(tensor, dim=self.dims, keepdim=True)
                    - torch.amin(tensor, dim=self.dims, keepdim=True)
                    + self.eps
                )
            else:
                raise NotImplementedError("normtype not implemented (scaler)")

        else:
            raise ValueError("should not happen")
        