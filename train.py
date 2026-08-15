from typing import overload
import torch
from numpy.lib import save
from util import Logger, accuracy, TotalMeter
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from util.prepossess import mixup_criterion, mixup_data
from util.loss import mixup_cluster_loss
from sklearn.metrics import roc_auc_score, confusion_matrix
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicTrain:

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        self.logger = Logger()
        self.model = model.to(device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = train_config['train']['epochs']
        self.optimizers = optimizers
        self.best_acc = 0
        self.best_model = None
        self.best_acc_val = 0
        self.best_auc_val = 0
        self.best_auc_test = 0
        self.best_sen = 0
        self.best_spe = 0
        self.best_f1 = 0
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.use_smri = train_config['train'].get('use_smri', False)

        self.group_loss = train_config['train']['group_loss']
        self.train_config = train_config
        self.sparsity_loss = train_config['train']['sparsity_loss']
        self.sparsity_loss_weight = train_config['train']['sparsity_loss_weight']
        self.save_path = log_folder

        self.save_learnable_graph = True

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy, \
            self.val_accuracy, self.test_accuracy, self.edges_num = [
            TotalMeter() for _ in range(7)]

        self.loss1, self.loss2, self.loss3 = [TotalMeter() for _ in range(3)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy, self.test_accuracy,
                      self.train_loss, self.val_loss, self.test_loss, self.edges_num,
                      self.loss1, self.loss2, self.loss3]:
            meter.reset()

    def train_per_epoch(self, optimizer):

        self.model.train()

        for data_in, pearson, label, _, smri_aseg, smri_destrieux, smri_wmparc, smri_tensor in self.train_dataloader:
            label = label.long()

            data_in, pearson, label, smri_aseg, smri_destrieux, smri_wmparc = data_in.to(
                device), pearson.to(device), label.to(device), smri_aseg.to(device), smri_destrieux.to(device), smri_wmparc.to(device)
            smri_tensor = smri_tensor.to(device)
            smri_encoder_type = self.train_config['data'].get(
                'smri_encoder',
                'mlp'
            )
            # inputs, nodes, targets_a, targets_b, lam, mixed_smri = mixup_data(
            #     data_in, pearson, label, 1, device,
            #     smri_list=[smri_aseg, smri_destrieux, smri_wmparc] if self.use_smri else None
            #     )

            if self.use_smri:
    
                if smri_encoder_type == 'gcn':

                    smri_for_mixup = [
                        smri_aseg,
                        smri_destrieux,
                        smri_wmparc
                    ]

                elif smri_encoder_type == 'mlp':

                    smri_for_mixup = smri_tensor

            else:

                smri_for_mixup = None
            
            inputs, nodes, targets_a, targets_b, lam, mixed_smri = \
                mixup_data(
                    data_in,
                    pearson,
                    label,
                    1,
                    device,
                    smri_data=smri_for_mixup
                )
            output, learnable_matrix, edge_variance = self.model(inputs, nodes, (mixed_smri))
            
            loss = 2 * mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            if self.group_loss:
                loss += mixup_cluster_loss(learnable_matrix,
                                           targets_a, targets_b, lam)

            if self.sparsity_loss:
                sparsity_loss = self.sparsity_loss_weight * \
                                torch.norm(learnable_matrix, p=1)
                loss += sparsity_loss

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
            self.edges_num.update_with_weight(edge_variance, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for data_in, pearson, label, _, smri_aseg, smri_destrieux, smri_wmparc,smri_tensor in dataloader:
            label = label.long()
            data_in, pearson, label, smri_aseg, smri_destrieux, smri_wmparc, smri_tensor = data_in.to(
                device), pearson.to(device), label.to(device), smri_aseg.to(device), smri_destrieux.to(device), smri_wmparc.to(device), smri_tensor.to(device)
            if not self.use_smri:
    
                smri_input = None
            elif self.train_config['data']['smri_encoder'] == 'gcn':
                smri_input = [smri_aseg, smri_destrieux, smri_wmparc]
                
            elif self.train_config['data']['smri_encoder'] == 'mlp':
                smri_input = smri_tensor   
            else:
                raise ValueError(
                    "Unknown sMRI encoder type"
                )
                        
            output, _, _ = self.model(data_in, pearson,smri_input)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        con_matrix = confusion_matrix(labels, result)
        return [auc] + list(metric), con_matrix

    def generate_save_learnable_matrix(self):
        learable_matrixs = []

        labels = []

        for data_in, nodes, label, _ ,smri_aseg, smri_destrieux, smri_wmparc, smri_tensor in self.test_dataloader:
            label = label.long()
            data_in, nodes, label, smri_aseg, smri_destrieux, smri_wmparc = data_in.to(
                device), nodes.to(device), label.to(device), smri_aseg.to(device), smri_destrieux.to(device), smri_wmparc.to(device)
            smri_tensor = smri_tensor.to(device)
            # smri_input = (
            #     [smri_aseg, smri_destrieux, smri_wmparc]
            #     if self.use_smri else None
            # )
            
            if not self.use_smri:
    
                smri_input = None

            elif self.train_config['data']['smri_encoder'] == 'gcn':

                smri_input = [
                    smri_aseg,
                    smri_destrieux,
                    smri_wmparc
                ]

            elif self.train_config['data']['smri_encoder'] == 'mlp':

                smri_input = smri_tensor
            
            
            
            _, learable_matrix, _ = self.model(data_in, nodes, smri_input)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results, txt):

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)
        with open(self.save_path / "training_info.txt", 'a', encoding='utf-8') as f:
            f.write(txt)
        torch.save(self.best_model.state_dict(), self.save_path/f"model_{self.best_acc}%.pt")

    def train(self):
        training_process = []
        txt = ''
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])
            val_result, _ = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result, con_matrix = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            # if self.best_acc <= self.test_accuracy.avg:
            #     self.best_acc = self.test_accuracy.avg
            #     self.best_model = self.model

            if (con_matrix[0][0] + con_matrix[1][0]) != 0:
                SEN = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[1][0])
            else:
                SEN = 0

            if (con_matrix[1][1] + con_matrix[0][1]) != 0:
                SPE = con_matrix[1][1] / (con_matrix[1][1] + con_matrix[0][1])
            else:
                SPE = 0
                
                
            if self.best_acc <= self.val_accuracy.avg:
                self.best_acc_val = self.val_accuracy.avg
                self.best_auc_val = val_result[0]
                self.best_model = self.model
                
                self.best_acc = self.test_accuracy.avg
                self.best_auc_test = test_result[0]
                self.best_sen = SEN
                self.best_spe = SPE
                self.best_f1 = test_result[-4]
                
                
                

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train ACC:{self.train_accuracy.avg: .3f}%',
                f'Val ACC:{self.val_accuracy.avg: .2f}%',
                f'Val AUC:{val_result[0]:.2f}',
                f'Test ACC:{self.test_accuracy.avg: .2f}%',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test SEN:{SEN:.4f}',
                f'Test SPE:{SPE:.4f}',
                f'Test F1:{test_result[-4]:.4f}',
            ]))

            txt += f'Epoch[{epoch}/{self.epochs}] '+f'Train Loss:{self.train_loss.avg: .3f} '+f'Train ACC:{self.train_accuracy.avg: .3f}% '+f'Val ACC:{self.val_accuracy.avg: .3f}% '+ f'Val AUC:{val_result[0]:.3f} '+f'Test ACC:{self.test_accuracy.avg: .3f}% '+f'Test AUC:{test_result[0]:.4f} '+f'Test SEN:{SEN:.4f} '+f'Test SPE:{SPE:.4f} '+f'Test F1:{test_result[-4]:.4f}'+'\n'

            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg]
                                    + val_result + test_result)
        now = datetime.now()
        date_time = now.strftime("%m-%d-%H-%M-%S")
        self.save_path = self.save_path/Path(f"{self.best_acc: .3f}%_{date_time}")
        self.logger.info(" | ".join([
            f'Best_ACC[{self.best_acc}]'
        ]))
        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process, txt)
        return {
            'acc': self.best_acc,
            'auc': self.best_auc_test,
            'sen': self.best_sen,
            'spe': self.best_spe,
            'f1': self.best_f1,
        }
        
        