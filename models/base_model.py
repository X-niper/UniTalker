import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for all models."""

    def __init__(self):
        super(BaseModel, self).__init__()
        # self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *x):
        """Forward pass logic.

        :return: Model output
        """
        raise NotImplementedError

    def freeze_model(self, do_freeze: bool = True):
        for param in self.parameters():
            param.requires_grad = (not do_freeze)

    def summary(self, logger, writer=None):
        """Model summary."""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size())
                      for p in model_parameters]) / 1e6  # Unit is Mega
        logger.info('===>Trainable parameters: %.3f M' % params)
        if writer is not None:
            writer.add_text('Model Summary',
                            'Trainable parameters: %.3f M' % params)
