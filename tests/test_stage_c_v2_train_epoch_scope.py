import importlib
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, src, attention_mask, labels):
        # Keep a gradient path so backward() works in _run_train_epoch.
        ce_loss = self.anchor * 0.0 + 0.5
        wpp_logits = torch.zeros((src.size(0), 3), device=src.device) + self.anchor * 0.0
        return ce_loss, wpp_logits


class TrainEpochScopeRegressionTest(unittest.TestCase):
    def _run_one_step(self, module_name: str) -> dict:
        module = importlib.import_module(module_name)
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        batch = {
            "src": torch.zeros((2, 4, 5), dtype=torch.float32),
            "mask": torch.ones((2, 4), dtype=torch.long),
            "labels_for_loss": torch.zeros((2, 3), dtype=torch.long),
            "text": ["hello", "world"],
        }
        loader = [batch]

        fake_wpp = torch.zeros((2, 3), dtype=torch.float32)
        with patch.object(module, "make_wpp_labels", return_value=fake_wpp):
            return module._run_train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                word_to_idx={"hello": 0},
                aux_weight=0.5,
                device=torch.device("cpu"),
                log_every=1,
                grad_accum_steps=1,
            )

    def test_train_stage_c_v2_train_epoch_logging_has_no_global_args_dependency(self) -> None:
        metrics = self._run_one_step("training.train_stage_c_v2")
        self.assertIn("total_loss", metrics)

    def test_train_stage_c_v2_fixed_train_epoch_logging_has_no_global_args_dependency(self) -> None:
        metrics = self._run_one_step("training.train_stage_c_v2_fixed")
        self.assertIn("total_loss", metrics)


if __name__ == "__main__":
    unittest.main()
