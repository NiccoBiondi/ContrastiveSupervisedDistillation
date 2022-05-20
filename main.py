import os
import argparse
import numpy as np
from datetime import datetime

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import transforms

from avalanche.benchmarks.datasets import CIFAR100
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks import nc_benchmark

from resnet import resnet32
from metric import RecallAtKPluginSingleTask, RecallAtKPluginAllTasksAndCheckpoint, MinibatchTripletLoss, \
    EpochTripletLoss, MinibatchCELoss, EpochCELoss, MinibatchKDLoss, EpochKDLoss, MinibatchContrastiveLoss, \
    EpochContrastiveLoss, MinibatchGlobalLoss, EpochGlobalLoss, MyEvalExpAccuracy
from trainer.strategies import ILFGIR_strategy


def train(args):
    root_folder = args.root_folder
    dataset = args.dataset
    number_of_exp = args.n_exp
    per_exp_classes = None
    task_label = False
    shuffle_classes_exp = False
    bias_classifier = False
    norm_weights_classsifier = True

    train_batch = args.batch_size
    eval_batch = args.batch_size
    train_epochs = args.epochs + 1  # to do the eval phase at the end
    eval_every = 2 # args.eval_int

    # load dataset and transforms    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_dset = CIFAR100(root=f"{root_folder}/data", train=True, download=True, transform=transform_train)
    test_dset = CIFAR100(root=f"{root_folder}/data", train=False, download=True, transform=transform_test)

    initial_num_classes = 100 // number_of_exp

    scenario = nc_benchmark(
        train_dataset=train_dset,
        test_dataset=test_dset,
        n_experiences=number_of_exp,
        per_exp_classes=per_exp_classes,
        task_labels=task_label,
        shuffle=-shuffle_classes_exp,
    )

    print("Scenario n class", scenario.n_classes)
    print("Scenario n class per exp:", scenario.n_classes_per_exp[0])

    if not os.path.exists(f"{root_folder}/saved_models"):
        os.mkdir(f"{root_folder}/saved_models")

    bestModelPath = f"{root_folder}/saved_models/bestModel_2T_RN32_CIFAR100_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.pth"
    model = resnet32(bias_classifier=bias_classifier, norm_weights_classifier=norm_weights_classsifier,
                     num_classes=initial_num_classes)

    # OPTIMIZER CREATION
    optim = Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler_lr = None

    # DEFINE  LOGGERS
    interactive_logger = InteractiveLogger()

    # DEFINE THE EVALUATION PLUGIN
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True),
        timing_metrics(minibatch=True, epoch=True, epoch_running=True),
        RecallAtKPluginSingleTask(),
        RecallAtKPluginAllTasksAndCheckpoint(bestModelPath),
        MinibatchTripletLoss(),
        MinibatchCELoss(),
        MinibatchKDLoss(),
        MinibatchGlobalLoss(),
        MinibatchContrastiveLoss(),
        EpochTripletLoss(),
        EpochCELoss(),
        EpochKDLoss(),
        EpochGlobalLoss(),
        EpochContrastiveLoss(),
        MyEvalExpAccuracy(),
        loggers=[interactive_logger],
    )

    cl_strategy = ILFGIR_strategy(
        model,
        None,
        optim,
        CrossEntropyLoss(),
        dataset=dataset,
        bestModelPath=bestModelPath,
        lr_scheduler=scheduler_lr,
        train_mb_size=train_batch,
        train_epochs=train_epochs,
        eval_mb_size=eval_batch,
        device=device,
        evaluator=eval_plugin,
        eval_every=eval_every
    )

    # TRAINING LOOP
    print('Starting...')
    results = []

    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        res = cl_strategy.train(experience, [scenario.test_stream[0:experience.current_experience + 1]])
        print('Training completed')
        print("Start evaluation on all past experiences")
        results.append(cl_strategy.eval(scenario.test_stream[0:experience.current_experience + 1]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Incremental Learning Image Retrieval')
    parser.add_argument('--root_folder', type=str, default="./",
                        help="root folder path. Default current directory (./)")
    parser.add_argument('--dataset', type=str, default="CIFAR100", help="dataset name")
    parser.add_argument('--epochs', type=int, default=800, help="number of training epochs for each task")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--eval_int', type=int, default=5, help="evaluation interval")
    parser.add_argument('--n_exp', type=int, default=2, help="total number of tasks")
    parser.add_argument('--container', action='store_true', help="whether using container (eg Docker)")
    args = parser.parse_args()
    print(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args)
