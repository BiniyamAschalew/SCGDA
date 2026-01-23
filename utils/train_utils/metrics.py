from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import torch



# creating a metric class for flexible metric calculation
class BaseMetric:

    def __init__(self, config: dict):
        self.metrics = config["expt"]["metrics"]
        self.verbose = config["expt"]["verbose"]

    def __call__(self, logits, labels, tag=""):

        logits = logits.detach()
        labels = labels.detach()
        preds = logits.argmax(axis=1)

        labels = self.to_numpy(labels)
        preds = self.to_numpy(preds)

        result = {}


        for metric in self.metrics:
            metric_function = METRIC_FUNCTIONS[metric]
            result[metric + tag] = metric_function(labels, preds)

        return result

    def to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor


def eval_macro_f1(label, pred):
    
    f1 = f1_score(y_true=label, y_pred=pred, average='macro')
    return f1

def eval_micro_f1(label, pred):

    f1 = f1_score(y_true=label, y_pred=pred, average='micro')
    return f1

def eval_accuracy(label, pred):

    acc = accuracy_score(y_true=label, y_pred=pred)
    return acc

def eval_roc_auc(label, pred):

    auc = roc_auc_score(y_true=label, y_score=pred)
    return auc


METRIC_FUNCTIONS = {
    "accuracy": eval_accuracy,
    "macro_f1": eval_macro_f1,
    "micro_f1": eval_micro_f1,
    "roc_auc": eval_roc_auc,
}
