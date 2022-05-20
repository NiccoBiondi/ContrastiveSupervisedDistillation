import torch
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation import Metric


class MyAccuracy(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self):
        """
        Initialize the metric
        """
        # self.mean_acc = 0.0
        self.mean_acc = []
        

    def update(self, mb_logits_out, true_y):
        """
        Update metric value 
        """
        true_y = torch.as_tensor(true_y)
        mb_logits_out = torch.as_tensor(mb_logits_out)

        _, predicted_y = torch.max(mb_logits_out.data, 1)
        total = true_y.size(0)
        correct = (predicted_y == true_y).sum().item()
        acc = correct/total

        self.mean_acc.append(acc)
        # if(self.mean_acc == 0.0):
        #     self.mean_acc=acc
        # else:
        #     self.mean_acc = (self.mean_acc+acc)/2


    def result(self) -> float:
        """
        Emit the metric result
        """
        return sum(self.mean_acc) / len(self.mean_acc)

    def reset(self):
        """
        Reset the metric value
        """
        self.mean_acc = []


class MyEvalExpAccuracy(PluginMetric[float]):
    """
    This metric plugin will return a `float` value after
    each evaluation epoch
    """

    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()
        self._metric = MyAccuracy()
        self.x_coord = 0

    def reset(self) -> None:
        """
        Reset the metric
        """
        self._metric.reset()

    def result(self) -> float:
        """
        Emit the result
        """
        return self._metric.result()

    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Update accuracy value after each iteration
        """
        self._metric.update(strategy.mb_output, strategy.mb_y)
        
    def before_eval_exp(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the accuracy before the next experience begins
        """
        self.reset()

    def after_eval_exp(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        """
        Emit the result at the end of experience
        """
        value = self.result()
        
        self.x_coord += 1 # increment x value
        name = "Eval_Accuracy_x_exp"
        return [MetricValue(self, name, value, self.x_coord)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        name = "Eval_Accuracy_x_exp"
        return name



    
