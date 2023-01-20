import warnings
import torch.nn as nn
import torch

from .CNN import CNN
from .RNN import BidirectionalGRU

class CRNN(nn.Module):
    def __init__(self,
                 n_in_channel=1,
                 n_class=10,
                 attention=True,
                 activation="glu",
                 dropout = 0.5,
                 conv_dropout=0.5,
                 train_cnn=True,
                 rnn_type="BGRU",
                 n_RNN_cell=128,
                 n_layers_rnn = 2,
                 dropout_rnn=0,
                 freeze_bn=False,
                 **kwargs):
        """Initialize a CRNN

        Args:
            n_in_channel (int, optional): number of expected input channel. Defaults to 1.
            n_class (int, optional): number of classes. Defaults to 10.
            attention (bool, optional): if True, an attention layer is added after the recurrent block. Defaults to True.
            activation (str, optional): activation function. Defaults to "glu".
            dropout (float, optional): dropout. Defaults to 0.5.
            dropout_conv (float, optional): dropout used in the CNN block. Defaults to 0.5.
            train_cnn (bool, optional): if False, CNN parameters are frozen. Defaults to True.
            rnn_type (str, optional): RNN type. Defaults to "BGRU".
            n_RNN_cell (int, optional): number of recurrent layers. Defaults to 128.
            n_layers_rnn (int, optional): number of recurrent layers. Defaults to 2.
            dropout_rnn (int, optional): dropout used in the RNN block. Defaults to 0.
            freeze_bn (bool, optional): if True, normalization parameters are frozen[description]. Defaults to False.
            **kwargs : keywords arguments for CNN
        """
        super().__init__()
        
        self.attention = attention
        self.freeze_bn = freeze_bn
        
        n_in_cnn = n_in_channel
        self.cnn = CNN(n_in_channel=n_in_cnn,
                       activation=activation,
                       conv_dropout=conv_dropout, **kwargs)
        #self.train_cnn = train_cnn
        
        # Freezing CNN parameters if specified
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        if rnn_type.lower() == "bgru":
            nb_in = self.cnn.nb_filters[-1]
            self.rnn = BidirectionalGRU(n_in=nb_in,
                                        n_hidden=n_RNN_cell,
                                        dropout=dropout_rnn,
                                        num_layers=n_layers_rnn)
        else:
            raise NotImplementedError("CRNN only supports BGRU for a recurrent block")
        
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, n_class)
        # for focal loss we init bias to -2
        torch.nn.init.constant_(self.dense.bias, -2)
        self.sigmoid = nn.Sigmoid()
        
        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, n_class)
            torch.nn.init.constant_(self.dense_softmax.bias, -2)
            self.softmax = nn.Softmax(dim=-1)
            
    def forward(self, x, pad_mask=None, return_logits=False):
        """Apply the CRNN to the input sequence

        Args:
            x (tensor): input sequence tensor
            pad_mask (tensor, optional): Used to mask parts of the sequence if needed. Defaults to None.
        Returns:
            out (tensor) : classification output tensor
        """

        # size : (batch_size, n_freq, n_frames)
        if len(x.shape) == 3:
            x = x.transpose(1, 2).unsqueeze(1)
            # size : (batch_size, 1, n_frames, n_freq)
        elif len(x.shape) == 4:
            x = x.transpose(2, 3)
        
        # conv features
        x = self.cnn(x)
        bs, chan, n_frames, n_freq = x.size()
        
        if n_freq != 1:
            warnings.warn(f"Output shape is : {(bs, n_frames, chan * n_freq)}, from {n_freq} staying freq")
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, n_frames, chan * n_freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1) # [bs, frames, chan]
        
        # rnn features
        x = self.rnn(x)
        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        if not return_logits:
            strong = self.sigmoid(strong)
        if self.attention:
            soft = self.dense_softmax(x) #[bs, frames, nclass]
            if pad_mask is not None:
                soft = soft.masked_fill(pad_mask.transpose(1, 2), 1e-30) # mask attention
            soft = self.softmax(soft)
            soft = torch.clamp(soft, min=1e-7, max=1)
            weak = (strong * soft).sum(1) / soft.sum(1)  # [bs, nclass]
        
        else:
            # weak = strong.mean(1)
            # Linear softmax is better suited
            weak = (strong * strong).sum(dim = 1) / strong.sum(dim = 1)        
        return strong.transpose(1, 2), weak
    
    
    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters

        Args:
            mode (bool, optional): Useful for default train(). Defaults to True.
        """
        super().train(mode)
        #? Maybe there's a better way
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False