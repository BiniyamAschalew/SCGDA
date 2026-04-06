import time

import torch
import torch.nn.functional as F

from  models.baselines.simgda.simgda import SimGDA
from  utils.train_utils.mmd import MMD


def _safe_mmd(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    if int(features_a.size(0)) <= 0 or int(features_b.size(0)) <= 0:
        return float("nan")
    return float(MMD(features_a, features_b).detach().cpu().item())


class TrackedSimGDA(SimGDA):
    """SimGDA variant that records per-epoch alignment and performance diagnostics."""

    def __init__(self, config):
        super(TrackedSimGDA, self).__init__(config)
        self.history_rows = []
        self.seed = int(config["expt"]["seed"])

    @staticmethod
    def _half_split_indices(num_items: int, *, seed: int, device: torch.device):
        if int(num_items) < 2:
            return None, None

        split = int(num_items) // 2
        if split <= 0 or split >= int(num_items):
            return None, None

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        perm = torch.randperm(int(num_items), generator=generator).to(device=device)
        return perm[:split], perm[split:]

    def _masked_micro_f1(self, logits, data, *, use_mask: bool, mask_name: str):
        masked_logits, masked_labels, masked_mask = self.mask_logits_and_labels(
            logits,
            data,
            use_mask=use_mask,
            mask_name=mask_name,
        )
        if int(masked_mask.sum().item()) <= 0:
            return float("nan")
        return float(self.metrics(masked_logits, masked_labels)["micro_f1"])

    def _within_domain_mmd(self, features: torch.Tensor, *, seed: int) -> float:
        left_idx, right_idx = self._half_split_indices(
            int(features.size(0)),
            seed=seed,
            device=features.device,
        )
        if left_idx is None or right_idx is None:
            return float("nan")
        return _safe_mmd(features[left_idx], features[right_idx])

    def _epoch_diagnostics(self, source_data, target_data, *, epoch: int, use_mask: bool):
        self.simgda.eval()

        with torch.no_grad():
            source_features = self.simgda.feat_bottleneck(source_data.x, source_data.edge_index)
            target_features = self.simgda.feat_bottleneck(target_data.x, target_data.edge_index)

            source_logits = self.simgda.feat_classifier(source_features, source_data.edge_index)
            target_logits = self.simgda.feat_classifier(target_features, target_data.edge_index)
            source_logits = F.log_softmax(source_logits, dim=1)
            target_logits = F.log_softmax(target_logits, dim=1)

        source_seed = self.seed + 1009 * int(epoch + 1) + 17
        target_seed = self.seed + 1009 * int(epoch + 1) + 53

        return {
            "source_micro_f1": self._masked_micro_f1(
                source_logits,
                source_data,
                use_mask=use_mask,
                mask_name="val_mask",
            ),
            "target_micro_f1": self._masked_micro_f1(
                target_logits,
                target_data,
                use_mask=use_mask,
                mask_name="val_mask",
            ),
            "source_target_mmd": _safe_mmd(source_features, target_features),
            "source_source_mmd": self._within_domain_mmd(source_features, seed=source_seed),
            "target_target_mmd": self._within_domain_mmd(target_features, seed=target_seed),
        }

    def fit(self, source_data, target_data, use_mask=False, oracle=False):
        source_loader = self.get_loader(source_data)
        target_loader = self.get_loader(target_data)

        self.simgda = self.init_model(**self.kwargs)
        self.history_rows = []

        optimizer = torch.optim.Adam(
            self.simgda.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        start_time = time.time()

        from tqdm import tqdm

        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0.0
            epoch_source_logits = None
            epoch_source_labels = None

            for sampled_source_data, sampled_target_data in zip(source_loader, target_loader):
                self.simgda.train()

                loss, source_logits, _ = self.forward_model(
                    sampled_source_data,
                    sampled_target_data,
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

            train_micro = float(self.metrics(epoch_source_logits, epoch_source_labels)["micro_f1"])
            diagnostics = self._epoch_diagnostics(
                source_data,
                target_data,
                epoch=epoch,
                use_mask=use_mask,
            )
            row = {
                "epoch": int(epoch + 1),
                "loss": float(epoch_loss),
                **diagnostics,
            }
            self.history_rows.append(row)

            self.log(
                epoch,
                epoch_loss,
                {
                    "train_source_micro_f1": train_micro,
                    **diagnostics,
                },
            )

        self.train_time = time.time() - start_time
        self.finish()
