import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim


class LogisticRegressionModel(pl.LightningModule):
    def __init__(self, input_dim, num_classes, learning_rate=0.001):
        super().__init__()
        self.model = nn.Linear(input_dim, num_classes)  # Logistic regression layer
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)  # Compute logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
