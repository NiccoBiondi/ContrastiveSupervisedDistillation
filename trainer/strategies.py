from typing import Optional, Sequence, List
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.plugins import StrategyPlugin, EvaluationPlugin
from avalanche.training.strategies import BaseStrategy
from avalanche.benchmarks.scenarios import Experience
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader

from trainer.plugins import ILFGIR_plugin


class ILFGIR_strategy(BaseStrategy):
    def __init__(self, model: Module, prev_model_frozen: Module, optimizer: Optimizer, criterion,
                 dataset, bestModelPath, lr_scheduler=None,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1,
                 lambda_kd=1., lambda_csd=1.):

        ILFGIR = ILFGIR_plugin(dataset, bestModelPath, lr_scheduler, prev_model_frozen,
                               lambda_kd, lambda_csd)
        if plugins is None:
            plugins = [ILFGIR]
        else:
            plugins.insert(0, ILFGIR)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def make_train_dataloader(self,
                              num_workers=0,
                              shuffle=True,
                              pin_memory=True):
        self.dataloader = TaskBalancedDataLoader(
            self.adapted_dataset,
            oversample_small_groups=True,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=False
        )

    def train_exp(self, experience: Experience, eval_streams=None, **kwargs):
        """ Training loop over a single Experience object.

        :param experience: CL experience information.
        :param eval_streams: list of streams for evaluation.
            If None: use the training experience for evaluation.
            Use [] if you do not want to evaluate during training.
        :param kwargs: custom arguments.
        """
        self.experience = experience
        self.model.train()

        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Sequence):
                eval_streams[i] = [exp]

        # Data Adaptation (e.g. add new samples/data augmentation)
        self._before_train_dataset_adaptation(**kwargs)
        self.train_dataset_adaptation(**kwargs)
        self._after_train_dataset_adaptation(**kwargs)
        self.make_train_dataloader(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        self.model = self.model_adaptation()

        self._before_training_exp(**kwargs)
        
        do_final = True
        if self.eval_every > 0 and \
                (self.train_epochs - 1) % self.eval_every == 0:
            do_final = False

        for _ in range(self.train_epochs):
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            self.training_epoch(**kwargs)
            self._after_training_epoch(**kwargs)
            self._periodic_eval(eval_streams, do_final=False)

        # Final evaluation
        self._periodic_eval(eval_streams, do_final=do_final)
        self._after_training_exp(**kwargs)
