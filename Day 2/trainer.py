# ============================================================
# trainer.py — Generic training / evaluation engine..
# ============================================================

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE, MODEL_PATH, RESULTS_DIR
from utils import ensure_dirs


class Trainer:
    """
    Encapsulates one full training experiment.

    Args:
        model      : any nn.Module
        optimizer  : pre-configured optimiser
        criterion  : loss function (CrossEntropyLoss for classification)
        device     : 'cuda' or 'cpu'

    Usage:
        trainer = Trainer(model, opt, criterion)
        history = trainer.fit(train_loader, val_loader, epochs=20)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = DEVICE,
    ) -> None:
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device

        # We track the best validation accuracy to save the best checkpoint.
        self._best_val_acc = 0.0
        ensure_dirs()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """One full pass over the training set."""
        self.model.train()   # activates Dropout, BatchNorm in train mode
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()       # reset gradients
            logits = self.model(images)      # forward
            loss   = self.criterion(logits, labels)
            loss.backward()                  # backward
            self.optimizer.step()            # update weights

            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """
        One full pass over a validation / test set 
        we don't need to calculate the gradient here
        """
        self.model.eval()    # deactivates Dropout, uses running stats in BN
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            loss   = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_best: bool = True,
    ) -> dict:
        """
        Train for epochs and return a history dictionary.

        Saving the best model not the last :
        the last epoch may have slightly worse generalisation than
        the epoch where the val curve peaked.
        """
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }

        print(f"\n{'='*60}")
        print(f"Training on {self.device.upper()} for {epochs} epoch(s)")
        print(f"{'='*60}")
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self._train_epoch(train_loader)
            vl_loss, vl_acc = self._eval_epoch(val_loader)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vl_acc)

            # Save checkpoint whenever val accuracy improves
            if save_best and vl_acc > self._best_val_acc:
                self._best_val_acc = vl_acc
                torch.save(self.model.state_dict(), MODEL_PATH)

            print(
                f"Epoch [{epoch:>3}/{epochs}]  "
                f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:5.1f}%  |  "
                f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:5.1f}%"
            )

        elapsed = time.time() - t0
        print(f"\nTraining complete in {elapsed:.1f}s")
        print(f"Best Val Acc: {self._best_val_acc:.1f}%")
        return history

    def overfit_single_sample(
        self,
        loader: DataLoader,
        epochs: int,
    ) -> dict:
        """
        Special loop for the Sanity Check (Step 1).

        We try to memorise ONE sample.
        Success criteria: training loss → ~0, training accuracy → 100%.
        If this fails, something is broken in the model or loss —
        fix it here before wasting GPU-hours on the full dataset.
        """
        history = {"train_loss": [], "train_acc": []}

        print(f"\n{'='*60}")
        print("STEP 1 — SANITY CHECK: Overfitting a SINGLE sample")
        print(f"{'='*60}")

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self._train_epoch(loader)
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)

            if epoch % 20 == 0 or epoch == 1:
                print(
                    f"  Epoch [{epoch:>3}/{epochs}]  "
                    f"Loss: {tr_loss:.6f}  Acc: {tr_acc:.1f}%"
                )

        final_loss = history["train_loss"][-1]
        passed = final_loss < 0.01
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"\nSanity Check {status}  (final loss = {final_loss:.6f})")
        return history