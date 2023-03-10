from email.base64mime import header_encode
import torch.nn as nn
import torch

from .RNN import BidirectionalGRU


class HearDetector(nn.Module):
    def __init__(
        self,
        hear_encoder="open_l3_6144",
        fine_tuning=False,
        n_class=10,
        attention=True,
        dropout=0.1,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_rnn=2,
        dropout_rnn=0,
        **kwargs,
    ):
        """Initialize a CRNN

        Args:
            n_in_channel (int, optional): number of expected input channel. Defaults to 1.
            n_class (int, optional): number of classes. Defaults to 10.
            attention (bool, optional): if True, an attention layer is added after the recurrent block. Defaults to True.
            dropout (float, optional): dropout. Defaults to 0.5.
            rnn_type (str, optional): RNN type. Defaults to "BGRU".
            n_RNN_cell (int, optional): number of recurrent layers. Defaults to 128.
            n_layers_rnn (int, optional): number of recurrent layers. Defaults to 2.
            dropout_rnn (int, optional): dropout used in the RNN block. Defaults to 0.
            **kwargs : keywords arguments for CNN
        """
        super().__init__()

        if hear_encoder == "open_l3_6144":
            self.embed_size = 6144
        elif hear_encoder == "open_l3_512":
            self.embed_size = 512
        # embedding size : melspec (48* F * 16 T (=768) + 1295 * 2 (windows 160ms & 800 ms))
        elif hear_encoder == "passt_2levelmel":
            self.embed_size = 3358
        elif hear_encoder == "passt_2level":
            self.embed_size = 2590
        elif hear_encoder == "passt_base":
            self.embed_size = 1295

        else:
            raise NotImplementedError(f"encoder {hear_encoder} is not supported")

        self.fine_tuning = fine_tuning
        self.rnn_type = rnn_type
        self.attention = attention

        if rnn_type.lower() == "bgru":
            self.rnn = BidirectionalGRU(
                n_in=self.embed_size,
                n_hidden=n_RNN_cell,
                dropout=dropout_rnn,
                num_layers=n_layers_rnn,
            )
            dim_dense = [n_RNN_cell * 2, n_class]

        elif rnn_type.lower() == "none":
            self.rnn = nn.Identity()
            dim_dense = [self.embed_size, n_class]

        elif rnn_type.lower() == "heareval":
            self.rnn = nn.Linear(self.embed_size, 1024)
            dim_dense = [1024, n_class]

        else:
            raise NotImplementedError(
                "Current implementation only supports BGRU for a recurrent block"
            )

        self.dense = nn.Linear(dim_dense[0], dim_dense[1])
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        if self.attention:
            self.dense_softmax = nn.Linear(dim_dense[0], dim_dense[1])
            self.softmax = nn.Softmax(dim=-1)

        if rnn_type.lower() == "none_2dense":
            self.dense = nn.Sequential(
                nn.Linear(self.embed_size, 1024), nn.Linear(1024 * 2, n_class)
            )
        if rnn_type.lower() == "heareval":
            # self.bn = nn.BatchNorm1d(1024)
            self.bn = nn.ReLU()

    def forward(self, x, pad_mask=None):
        """Apply the classification head to the hear embedding input

        Args:
            x (tensor): input sequence tensor
            pad_mask (tensor, optional): Used to mask parts of the sequence if needed. Defaults to None.
        Returns:
            out (tensor) : classification output tensor
        """
        # rnn features
        x = self.rnn(x)
        x = self.dropout(x)
        if self.rnn_type == "heareval":
            x_t = x.transpose(1, 2)
            strong = self.dense(self.bn(x_t).transpose(1, 2))
        else:
            strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            soft = self.dense_softmax(x)  # [bs, frames, nclass]
            if pad_mask is not None:
                soft = soft.masked_fill(
                    pad_mask.transpose(1, 2), 1e-30
                )  # mask attention
            soft = self.softmax(soft)
            soft = torch.clamp(soft, min=1e-7, max=1)
            weak = (strong * soft).sum(1) / soft.sum(1)  # [bs, nclass]

        else:
            # weak = strong.mean(1)
            # Linear softmax is better suited than mean
            weak = (strong * strong).sum(dim=1) / strong.sum(dim=1)

        return strong.transpose(1, 2), weak
