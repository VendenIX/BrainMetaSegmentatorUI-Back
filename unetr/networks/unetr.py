# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module only contains the U-Net Transformer network."""

from typing import Any, List, OrderedDict, Tuple, Union

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT
import torch
import torch.nn as nn
import pytorch_lightning as pl

class UNETR(pl.LightningModule):
    """
    Network that is similar to a U-Net for image segmentation with
    an adaptation to use transformers and their attention mecanism.

    Attributes:
        num_layers: Number of layers in the Vision Transformer.
        out_channels: Number of output channels.
        patch_size: Size of the patch (tuple of `feature_size`) for the embedding in the transformer.
        feat_size: Number of patches that can be put in a single image.
        feature_size: Size of the feature.
        hidden_size: Dimension of hidden layer.
        classification: Boolean that represents if we are in a classification problem in the Vision Transformer.
        vit: Vision Transformer block.
        encoder1: First downsampling block (linked to the hidden states of the `vit`).
        encoder2: Second downsampling block (linked to the hidden states of the `vit`).
        encoder3: Third downsampling block (linked to the hidden states of the `vit`).
        encoder4: Fourth downsampling block (linked to the hidden states of the `vit`).
        decoder5: First upsampling block (linked to `vit` and `encoder4` outputs).
        decoder4: Second upsampling block (linked to `encoder3` and `decoder5` outputs).
        decoder3: Third upsampling block (linked to `encoder2` and `decoder4` outputs).
        decoder2: Fourth upsampling block (linked to `encoder1` and `decoder3` outputs).
        out: Output block (take only `decoder2` output).

    References:
        "Hatamizadeh et al., UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> "UNETR":
        """
        Arguments:
            in_channels: Dimension of input channels.
            out_channels: Dimension of output channels.
            img_size: Dimension of input image.
            feature_size: Dimension of network feature size.
            hidden_size: Dimension of hidden layer.
            mlp_dim: Dimension of feedforward layer.
            num_heads: Number of attention heads.
            pos_embed: Position embedding layer type.
            norm_name: Feature normalization type and arguments.
            conv_block: Bool argument to determine if convolutional block is used.
            res_block: Bool argument to determine if residual block is used.
            dropout_rate: Fraction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        Raises:
            AssertionError: when dropout_rate is not between 0 and 1 or
                hidden_size is not divisible by num_heads (needed for transformer blocks).
            KeyError: when a wrong value of pos_embed is passed.
        """

        super().__init__()
        self.save_hyperparameters()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.out_channels = out_channels
        self.patch_size = (feature_size, feature_size, feature_size)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_state_dict: OrderedDict[str, torch.Tensor],
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        new_out_channels: int = 0,
        number_of_blocks_to_tune: int = 0,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> "UNETR":
        """Load networks weight from a pretrained model.
        
        In this method, we can easily perform a transformation of the network to make 
        a finetuning (modify last layer and reinitialize multiple blocks weights).

        Arguments:
            pretrained_model_state_dict: State dict of the pretrained model (need to be separetaly load).
            in_channels: Dimension of input channels.
            out_channels: Dimension of output channels.
            img_size: Dimension of input image.
            new_out_channels: Dimension of the new output channels (for finetuning).
            number_of_blocks_to_tune: Number of blocks to tune (for finetuning).
            feature_size: Dimension of network feature size.
            hidden_size: Dimension of hidden layer.
            mlp_dim: Dimension of feedforward layer.
            num_heads: Number of attention heads.
            pos_embed: Position embedding layer type.
            norm_name: Feature normalization type and arguments.
            conv_block: Bool argument to determine if convolutional block is used.
            res_block: Bool argument to determine if residual block is used.
            dropout_rate: Fraction of the input units to drop.
        
        Raises:
            AssertionError: 
                - When `new_out_channels` is positive but `number_of_blocks_to_tune`
                is not positive (cannot change last block if we doesn't want to tune any block).
                - When `number_of_blocks_to_tune` is greater than 10, because there are only 10 blocks.
        """
        if new_out_channels > 0:
            assert number_of_blocks_to_tune > 0, "To change the last block, you need to authorize to tune it. Please choose a positive value (0 excluded)"

        if number_of_blocks_to_tune > 0:
            assert number_of_blocks_to_tune <= 10, "Too much block to tune. Please choose a number between 0 and 10 included"

        # creation and model loading
        model = cls(in_channels, out_channels, img_size, feature_size=feature_size,
                    hidden_size=hidden_size, mlp_dim=mlp_dim, num_heads=num_heads,
                    pos_embed=pos_embed, norm_name=norm_name, conv_block=conv_block,
                    res_block=res_block, dropout_rate=dropout_rate)
        model.load_state_dict(pretrained_model_state_dict)

        # finetuning of the model
        if number_of_blocks_to_tune > 0:
            model.number_of_blocks_to_tune = number_of_blocks_to_tune
            
            # change number of output channels
            if out_channels != new_out_channels:
                model.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=new_out_channels)
                model.out_channels = new_out_channels
            
            # reinitialize all blocks to tune
            model.reinit_weights()

        return model
    
    @property
    def backbone(self) -> List[nn.Module]:
        """Returns the part of the network that corresponding to
        the backbone network to reuse for finetuning.
        
        Returns:
            blocks: Network parts in a list.
        
        Raises:
            AttributeError: Raised when `number_of_blocks_to_tune` attribute is undefined,
                in other words, when we are not in a finetuning.
        
        See also:
            _get_blocks
        """
        if not hasattr(self, "number_of_blocks_to_tune"):
            raise AttributeError("you're not in fintuning the model")
        
        return self._get_blocks()
    
    def _get_blocks(self, to_not_tune: bool = True) -> List[nn.Module]:
        """Gets blocks of the network.

        If to_not_tune is activated, only the blocks that we doesn't want to
        tune will be returned. Else, all other ones.
        
        Arguments:
            to_not_tune: Represents the fact that the method will return blocks to tune or not.
        
        Returns:
            blocks: Network blocks according to arguments.
        """
        blocks = [self.vit, self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.decoder5, self.decoder4, self.decoder3, self.decoder2, self.out]

        # get only blocks to not tune
        if to_not_tune:
            return blocks[:len(blocks) - self.number_of_blocks_to_tune]
        
        # get blocks we want to tune
        if self.number_of_blocks_to_tune <= 0:
            return blocks
        return blocks[len(blocks) - self.number_of_blocks_to_tune:]
    
    def reinit_weights(self) -> None:
        """Reinitializes the parameters weights of the right part
        of the network/model following distributions.

        You can view the association between layer types and distributions below:
        - for the filters in convolutional layers, we use the Kaiming uniform initializer [1];
        - for the biases in convolutional layers, we reinit to zeros.

        References:
            [1] "He et al., Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <https://arxiv.org/abs/1502.01852>"
        """
        for block in self._get_blocks(to_not_tune=False):
            for name, param in block.named_parameters():
                if "conv.weight" in name:
                    torch.nn.init.kaiming_uniform_(param) # better than Xavier or default values
                elif "conv.bias" in name:
                    torch.nn.init.zeros_(param)

    def proj_feat(self, x: torch.Tensor, hidden_size: int, feat_size: Tuple[int, int, int]) -> torch.Tensor:
        """Computes a feature projection.

        The goal of this method is to change the way that we have to see
        the `x` tensor by changing its dimensions. A permutation of axis is
        realized to put temporal dimension second after batch and before
        slices.
        
        Arguments:
            x: Tensor to project.
            hidden_size: Output size of the hidden layer.
            feat_size: Size of the feature.

        Returns:
            x: New view of `x` tensor.
        """
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """Realizes the forward to make prediction.
        
        Arguments:
            x_in: Tensor data to predict.
        
        Returns:
            logits: Predictions tensor.
        """
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits

    def print_parameters(self, **print_kwargs: Any) -> None:
        """Prints in the console all the network parameters.
        
        All the associated names and tensor parameters
        are printed to the console to check the parameters
        sizes or values.

        Arguments:
            print_kwargs: Keyword arguments to pass to the print function.
        """
        for name, params in self.named_parameters():
            print(name, params.size(), params, **print_kwargs)
