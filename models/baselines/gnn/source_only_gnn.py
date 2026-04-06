import time

import torch
import torch.nn.functional as F

from  models.baselines.gnn.gnn import GNN
from  models.baselines.gnn.source_only_gnn_base import SourceOnlyGNNBase


def _source_only_nll_loss(logits, labels, mask):
    if int(mask.sum().item()) <= 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return F.nll_loss(logits[mask], labels[mask])


class SourceOnlyGNN(GNN):
    """Source-only baseline that never consumes target batches during training."""

    def init_model(self, **kwargs):
        model = SourceOnlyGNNBase(self.config).to(self.device)
        return model

    def forward_model(self, source_data, target_data=None, use_mask=False, oracle=False):
        source_logits = self.gnn(source_data.x, source_data.edge_index)
        source_mask = self.get_mask(source_data, use_mask=use_mask, mask_name="train_mask")
        loss = _source_only_nll_loss(source_logits, source_data.y, source_mask)
        return loss, source_logits

    def fit(self, source_data, target_data=None, use_mask=False, oracle=False):
        source_loader = self.get_loader(source_data)

        self.gnn = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.gnn.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        start_time = time.time()

        from tqdm import tqdm

        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0.0
            epoch_source_logits = None
            epoch_source_labels = None

            for sampled_source_data in source_loader:
                sampled_source_data = sampled_source_data.to(self.device)
                self.gnn.train()

                loss, source_logits = self.forward_model(
                    sampled_source_data,
                    use_mask=use_mask,
                    oracle=oracle,
                )
                epoch_loss += float(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_source_logits, train_source_labels, train_source_mask = self.mask_logits_and_labels(
                    source_logits,
                    sampled_source_data,
                    use_mask=use_mask,
                    mask_name="train_mask",
                )
                if int(train_source_mask.sum().item()) > 0:
                    if epoch_source_logits is None:
                        epoch_source_logits = train_source_logits
                        epoch_source_labels = train_source_labels
                    else:
                        epoch_source_logits = torch.cat((epoch_source_logits, train_source_logits))
                        epoch_source_labels = torch.cat((epoch_source_labels, train_source_labels))

            if epoch_source_logits is None or epoch_source_labels is None:
                raise RuntimeError("No source training nodes were available for loss/metric computation.")

            train_results = self.metrics(epoch_source_logits, epoch_source_labels)
            self.log(epoch, epoch_loss, train_results)

        self.train_time = time.time() - start_time
        self.finish()
