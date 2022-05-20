import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
import torch
from losses import SupConLoss, kd_loss
from torch.nn import CrossEntropyLoss

from pytorch_metric_learning import losses, miners, distances


class ILFGIR_plugin(StrategyPlugin):
    def __init__(self, dataset, bestModelPath, 
                 scheduler_lr, prev_model_frozen,
                 lambda_kd, lambda_csd):  
        super().__init__()

        self.dataset = dataset

        self.global_loss = torch.tensor(0.0)
        self.ceLoss = torch.tensor(0.0)
        self.tripletLoss = torch.tensor(0.0)
        self.kdLoss = torch.tensor(0.0)

        self.contrastiveLoss = torch.tensor(0.0)

        self.lambda_kd = lambda_kd
        self.lambda_csd = lambda_csd

        self.bestModelPath = bestModelPath
        self.scheduler_lr = scheduler_lr
        self.prev_model_frozen = prev_model_frozen

        self.mb_out_flattened_features = None
        self.out_logits = None
        self.Triplet_Loss = losses.TripletMarginLoss(distance=distances.CosineSimilarity(), margin=1)
        self.CELoss = CrossEntropyLoss()
        self.miner = miners.BatchHardMiner()
        self.Contrastive = SupConLoss(contrast_mode="one")

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        self.mb_out_flattened_features = strategy.mb_output[1]
        strategy.mb_output = strategy.mb_output[0]

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        if self.prev_model_frozen is None:
            self.ceLoss = strategy.loss

            hard_pairs = self.miner(self.mb_out_flattened_features, strategy.mb_y)
            self.tripletLoss = self.Triplet_Loss(F.normalize(self.mb_out_flattened_features), 
                                                 strategy.mb_y, hard_pairs)

            strategy.loss = self.ceLoss + self.tripletLoss
            self.global_loss = strategy.loss
        else:
            frozen_prev_model_logits, frozen_prev_model_flattened_features = self.prev_model_frozen(strategy.mb_x)

            hard_pairs = self.miner(self.mb_out_flattened_features, strategy.mb_y)
            self.tripletLoss = self.Triplet_Loss(F.normalize(self.mb_out_flattened_features), strategy.mb_y,
                                                     hard_pairs)
            self.contrastiveLoss = self.Contrastive(torch.cat(
                                                    [F.normalize(self.mb_out_flattened_features).unsqueeze(1),
                                                     F.normalize(frozen_prev_model_flattened_features).unsqueeze(1)
                                                    ], dim=1), strategy.mb_y)

            n_old_class = frozen_prev_model_logits.shape[1]
            mb_old_class_logits = strategy.mb_output[:, :n_old_class]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            ce_weights = torch.zeros(strategy.mb_output.shape[1]).to(device)
            ce_weights[n_old_class:] = 1
            cross_entropy_loss = CrossEntropyLoss(weight=ce_weights)
            self.ceLoss = cross_entropy_loss(strategy.mb_output, strategy.mb_y)

            n_new_images = strategy.mb_x.shape[0]

            self.kdLoss = kd_loss(n_new_images, mb_old_class_logits, frozen_prev_model_logits)

            strategy.loss = self.ceLoss + self.tripletLoss + self.kdLoss + self.contrastiveLoss 
            self.global_loss = strategy.loss

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        if (self.scheduler_lr != None):
            self.scheduler_lr.step()

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        if (strategy.experience.current_experience == 0):
            strategy.optimizer = Adam(strategy.model.parameters(), lr=0.001, weight_decay=5e-4)
        else:
            strategy.optimizer = Adam(strategy.model.parameters(), lr=0.00001, weight_decay=5e-4)

        self.scheduler_lr = MultiStepLR(strategy.optimizer, milestones=[200, 400, 600], gamma=0.1) 


    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        checkpoint = torch.load(self.bestModelPath)
        self.prev_model_frozen = copy.deepcopy(strategy.model)
        self.prev_model_frozen.load_state_dict(checkpoint['model_state_dict'])
        strategy.model.load_state_dict(checkpoint['model_state_dict'])

    def after_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        self.mb_out_flattened_features = strategy.mb_output[1]
        strategy.mb_output = strategy.mb_output[0]
