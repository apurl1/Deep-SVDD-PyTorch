from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import logging
import time
import torch
import torch.cuda
import torch.optim as optim
import numpy as np


class AETrainerWheel(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get train data loader
        train_loader = dataset.train_set

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for inputs, labels in train_loader:
                if (len(inputs) == 32):
                    inputs = inputs.to(self.device)
                    # inputs = np.expand_dims(inputs, 1)
                    inputs = inputs.unsqueeze(1) 
                    # Zero the network parameter gradients
                    optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                    outputs = ae_net(inputs.float())
                    # print(outputs.shape, inputs.shape)
                    scores = torch.sum((outputs.float() - inputs.float()) ** 2, dim=tuple(range(1, outputs.dim())))
                    loss = torch.mean(scores)
                    loss.backward()
                    optimizer.step()

                    loss_epoch += loss.item()
                    n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        pretrain_time = time.time() - start_time
        logger.info('Pretraining time: %.3f' % pretrain_time)
        logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        test_loader = dataset.test_set

        # Testing
        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                if len(inputs) == 32:
                    inputs = inputs.to(self.device)
                    inputs = inputs.unsqueeze(1)
                    outputs = ae_net(inputs.float())
                    scores = torch.sum((outputs.float() - inputs.float()) ** 2, dim=tuple(range(1, outputs.dim())))
                    loss = torch.mean(scores)

                    # Save triple of (idx, label, score) in a list
                    idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                                scores.cpu().data.numpy().tolist()))

                    loss_epoch += loss.item()
                    n_batches += 1

        logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        test_auc = auc(fpr, tpr)
        logger.info('Test set AUC: {:.2f}%'.format(100. * test_auc))

        test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' % test_time)
        logger.info('Finished testing autoencoder.')
