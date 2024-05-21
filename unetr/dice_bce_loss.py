from functools import partial
from typing import Callable, Optional

from monai.losses.dice import DiceLoss
from monai.networks.utils import one_hot
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class DiceBCELoss(_Loss):
    """
    Compute both Dice loss and Binary Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Binary Cross Entropy Loss is shown in ``torch.nn.functional.binary_cross_entropy``.
    In this implementation, two deprecated parameters ``size_average`` and ``reduce`` are not supported.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        bce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
    ) -> None:
        """
        Arguments:
            ``bce_weight`` and ``lambda_bce`` are only used for binary cross entropy loss.
            ``reduction`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `binary_cross_entropy`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `binary_cross_entropy`.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example: `other_act = torch.tanh`.
                only used by the `DiceLoss`, don't need to specify activation function for `binary_cross_entropy`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from binary cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            bce_weight: a rescaling weight given to each class for binary cross entropy loss.
                See ``torch.nn.binary_cross_entropy()`` for more information.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_bce: the trade-off weight value for binary cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.binary_cross_entropy = partial(
            F.binary_cross_entropy_with_logits,
            weight=bce_weight,
            reduction=reduction,
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_bce < 0.0:
            raise ValueError("lambda_bce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.to_onehot_y = to_onehot_y
    
    def __repr__(self):
        return "DiceBCELoss"
    
    def __str__(self) -> str:
        return super().__repr__()

    def bce(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute BinaryCrossEntropy loss for the logits input and target.
        Will remove the channel dim according to PyTorch BinaryCrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html.
        
        Arguments:
            input_: the shape should be BNH[WD] and should to be logits and not probabilities.
            target: the shape should be BNH[WD] or B1H[WD].
        """
        assert input_.shape[1] != 1, "We need to have the logits instead of probabilities"

        if self.to_onehot_y and target.shape[1] == 1:
            target = one_hot(target, input_.shape[1], dim=1)

        target = target.float()
        return self.binary_cross_entropy(input_, target)

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            input_: the shape should be BNH[WD] and should to be logits and not probabilities.
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for `input` and `target` are different or 
                when number of channels for `target` is neither 1 nor the same as `input`.

        """
        if len(input_.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input_ and target should be the same.")

        dice_loss = self.dice(input_, target)
        bce_loss = self.bce(input_, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_bce * bce_loss

        return total_loss
