import sys
import tensorflow as tf
from tqdm import tqdm
import json
import torch
from torch import nn
import numpy as np
import contextlib
import os 

sys.path.append('../')

from capreolus.trainer import Trainer
from capreolus.trainer.tensorflow import TensorflowTrainer
from capreolus.trainer.pytorch import PytorchTrainer
from capreolus_extensions.LCEReranker_and_Loss import KerasLCEModel, TFLCELoss
from capreolus import ConfigOption
from capreolus.reranker.common import pair_hinge_loss, pair_softmax_loss
from capreolus import ConfigOption, get_logger
from capreolus import Searcher

logger = get_logger(__name__) 

@Trainer.register
class LCETensorflowTrainer(TensorflowTrainer):
    module_name = "LCEtensorflow"
    config_spec = TensorflowTrainer.config_spec + [
        ConfigOption("disable_position", False, "Whether to disable the positional embedding"),
        ConfigOption("disable_segment", False, "Whether to disable the segment embedding"),
    ]
    config_keys_not_in_path = ["fastforward", "boardname", "usecache", "tpuname", "tpuzone", "storage"]
    
    def get_loss(self, loss_name):
        try:
            if loss_name == "pairwise_hinge_loss":
                loss = TFPairwiseHingeLoss(reduction=tf.keras.losses.Reduction.NONE)
            elif loss_name == "crossentropy":
                loss = TFCategoricalCrossEntropyLoss(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            elif loss_name == "lce":
                loss = TFLCELoss(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            else:
                loss = tf.keras.losses.get(loss_name)
        except ValueError:
            loss = tf.keras.losses.get(loss_name)

        return loss

    def get_wrapped_model(self, model):
        if self.config["loss"] == "crossentropy":
            return KerasPairModel(model)
        if self.config["loss"] == "lce":
            return KerasLCEModel(model)

        return KerasTripletModel(model)

@Trainer.register
class LCEptTrainer(PytorchTrainer):
    module_name = "LCEpt"
    config_spec = PytorchTrainer.config_spec + [
        ConfigOption("disable_position", False, "Whether to disable the positional embedding"),
        ConfigOption("disable_segment", False, "Whether to disable the segment embedding"),
        ConfigOption("nneg", 3, "Number of negative samples to include"),
    ]

    # to avoid file name too long error
    config_keys_not_in_path = ["boardname", "softmaxloss", "fastforward", "gradacc", "warmupiters", "multithread"]
    
    
    def lce_loss(self, pos_neg_scores, *args, **kwargs):
        batch_size = self.config["batch"]
        nneg = self.config["nneg"]

        logsoftmax = nn.LogSoftmax(dim=-1)

        pos_neg_scores[0] = pos_neg_scores[0].view(
                1,
                batch_size,
                2
        )
        pos_neg_scores[0] = logsoftmax(pos_neg_scores[0])[:,:,-1]

        pos_neg_scores[1] = pos_neg_scores[1].view(
                nneg,
                batch_size,
                2
        )
        pos_neg_scores[1] = logsoftmax(pos_neg_scores[1])[:,:,-1]

        pos_neg_scores = torch.cat(pos_neg_scores, dim=0).transpose(0, 1)
        target_label = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        cross_entropy_lce = nn.CrossEntropyLoss(reduction='mean')
        loss_lce = cross_entropy_lce(pos_neg_scores, target_label)

        return loss_lce

    def single_train_iteration(self, reranker, train_dataloader):
        """Train model for one iteration using instances from train_dataloader.

        Args:
           model (Reranker): a PyTorch Reranker
           train_dataloader (DataLoader): a PyTorch DataLoader that iterates over training instances

        Returns:
            float: average loss over the iteration

        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = self.lce_loss

        iter_loss = []
        batches_since_update = 0
        batches_per_step = self.config["gradacc"]

        for bi, batch in tqdm(enumerate(train_dataloader), desc="Training iteration", total=self.n_batch_per_iter):
            
            # to avoid RuntimeError: Input, output and indices must be on the current device
            #batch = {k: v.to(self.device) if not isinstance(v, list) else v for k, v in batch.items()}
            new_batch = dict()
            for k, v in batch.items():
                if isinstance(v, list):
                    if isinstance(v[0], torch.Tensor):
                        relevant_tensor = v[0]
                        if len(v) > 0:
                            relevant_tensor = torch.stack(v)
                        new_batch[k] = relevant_tensor.to(self.device)    
                    else:
                        new_batch[k] = v    
                else:
                    new_batch[k] = v.to(self.device)

            batch = new_batch

            with self.amp_train_autocast():
                doc_scores = reranker.score(batch)
                loss = self.loss(doc_scores)

            iter_loss.append(loss)
            loss = self.scaler.scale(loss) if self.scaler else loss
            loss.backward()

            batches_since_update += 1
            if batches_since_update == batches_per_step:
                batches_since_update = 0
                # REF-TODO: save scaler state
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            if (bi + 1) % self.n_batch_per_iter == 0:
                # REF-TODO: save scheduler state along with optimizer
                self.lr_scheduler.step()
                break

        return torch.stack(iter_loss).mean()
    def predict(self, reranker, pred_data, pred_fn):
        """Predict query-document scores on `pred_data` using `model` and write a corresponding run file to `pred_fn`

        Args:
           model (Reranker): a PyTorch Reranker
           pred_data (IterableDataset): data to predict on
           pred_fn (Path): path to write the prediction run file to

        Returns:
           TREC Run

        """

        if self.config["amp"] in ("both", "pred"):
            self.amp_pred_autocast = torch.cuda.amp.autocast
        else:
            self.amp_pred_autocast = contextlib.nullcontext

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # save to pred_fn

        model = torch.nn.DataParallel(reranker.model).to(self.device)
        model.eval()

        preds = {}
        evalbatch = self.config["evalbatch"] if self.config["evalbatch"] > 0 else self.config["batch"]
        num_workers = 1 if self.config["multithread"] else 0
        pred_dataloader = torch.utils.data.DataLoader(pred_data, batch_size=evalbatch, pin_memory=True, num_workers=num_workers)
        with torch.autograd.no_grad():
            for batch in tqdm(pred_dataloader, desc="Predicting", total=len(pred_data) // evalbatch):
                if len(batch["qid"]) != evalbatch:
                    batch = self.fill_incomplete_batch(batch, batch_size=evalbatch)

                #batch = {k: v.to(self.device) if not isinstance(v, list) else v for k, v in batch.items()}
                new_batch = dict()
                for k, v in batch.items():
                    if isinstance(v, list):
                        if isinstance(v[0], torch.Tensor):
                            relevant_tensor = v[0]
                            if len(v) > 0:
                                relevant_tensor = torch.stack(v)
                            new_batch[k] = relevant_tensor.to(self.device)
                        else:
                            new_batch[k] = v      
                    else:
                        new_batch[k] = v.to(self.device)

                batch = new_batch
                
                with self.amp_pred_autocast():
                    scores = reranker.test(batch)
                scores = scores.view(-1).cpu().numpy().astype(np.float16)
                for qid, docid, score in zip(batch["qid"], batch["posdocid"], scores):
                    # Need to use float16 because pytrec_eval's c function call crashes with higher precision floats
                    preds.setdefault(qid, {})[docid] = score.item()

        os.makedirs(os.path.dirname(pred_fn), exist_ok=True)
        Searcher.write_trec_run(preds, pred_fn)

        return preds