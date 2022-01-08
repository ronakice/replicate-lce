import sys
import tensorflow as tf
from tqdm import tqdm
import json
import torch
from torch import nn
import numpy as np
import contextlib
import os 
import time

sys.path.append('../')

from capreolus.trainer import Trainer
from capreolus.trainer.tensorflow import TensorflowTrainer
from capreolus.trainer.pytorch import PytorchTrainer
from capreolus_extensions.LCEReranker_and_Loss import KerasLCEModel, TFLCELoss
from capreolus.reranker.common import pair_hinge_loss, pair_softmax_loss
from capreolus import ConfigOption, Searcher, constants, get_logger
from capreolus.utils.trec import convert_metric
from capreolus.evaluator import log_metrics_verbose, format_metrics_string

from torch.utils.tensorboard import SummaryWriter

RESULTS_BASE_PATH = constants["RESULTS_BASE_PATH"]
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
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


    def train(self, reranker, train_dataset, train_output_path, dev_data, dev_output_path, qrels, metric, benchmark):
        """Train a model following the trainer's config (specifying batch size, number of iterations, etc).

        Args:
           train_dataset (IterableDataset): training dataset
           train_output_path (Path): directory under which train_dataset runs and training loss will be saved
           dev_data (IterableDataset): dev dataset
           dev_output_path (Path): directory where dev_data runs and metrics will be saved

        """
        # Set up logging
        # TODO why not put this under train_output_path?
        summary_writer = SummaryWriter(RESULTS_BASE_PATH / "runs" / self.config["boardname"], comment=train_output_path)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        reranker.model = torch.nn.DataParallel(reranker.model, device_ids=range(torch.cuda.device_count()))
        reranker.model.to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, reranker.model.parameters()), lr=self.config["lr"])

        if self.config["amp"] in ("both", "train"):
            self.amp_train_autocast = torch.cuda.amp.autocast
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.amp_train_autocast = contextlib.nullcontext
            self.scaler = None

        # REF-TODO how to handle interactions between fastforward and schedule? --> just save its state
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda epoch: self.lr_multiplier(step=epoch * self.n_batch_per_iter)
        )

        dev_best_weight_fn, weights_output_path, info_output_path, loss_fn, metric_fn = self.get_paths_for_early_stopping(
            train_output_path, dev_output_path
        )

        num_workers = 1 if self.config["multithread"] else 0
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["batch"], pin_memory=True, num_workers=num_workers
        )

        # if we're fastforwarding, set first iteration and load last saved weights
        initial_iter, metrics = (
            self.fastforward_training(reranker, weights_output_path, loss_fn, metric_fn)
            if self.config["fastforward"]
            else (0, {})
        )
        dev_best_metric = metrics.get(metric, -np.inf)
        logger.info("starting training from iteration %s/%s", initial_iter + 1, self.config["niters"])
        logger.info(f"Best metric loaded: {metric}={dev_best_metric}")

        train_loss = []
        # are we resuming training? fastforward loss and data if so
        if initial_iter > 0:
            train_loss = self.load_loss_file(loss_fn)

            # are we done training? if not, fastforward through prior batches
            if initial_iter < self.config["niters"]:
                logger.debug("fastforwarding train_dataloader to iteration %s", initial_iter)
                self.exhaust_used_train_data(train_dataloader, n_batch_to_exhaust=initial_iter * self.n_batch_per_iter)

        logger.info(self.get_validation_schedule_msg(initial_iter))
        train_start_time = time.time()
        for niter in range(initial_iter, self.config["niters"]):
            niter = niter + 1  # index from 1
            reranker.model.train()

            iter_start_time = time.time()
            iter_loss_tensor = self.single_train_iteration(reranker, train_dataloader)
            logger.info("A single iteration takes {}".format(time.time() - iter_start_time))
            train_loss.append(iter_loss_tensor.item())
            logger.info("iter = %d loss = %f", niter, train_loss[-1])

            # save model weights only when fastforward enabled
            if self.config["fastforward"]:
                weights_fn = weights_output_path / f"{niter}.p"
                reranker.save_weights(weights_fn, self.optimizer)

            # predict performance on dev set
            if niter % self.config["validatefreq"] == 0:
                pred_fn = dev_output_path / f"{niter}.run"
                preds = self.predict(reranker, dev_data, pred_fn)

                # log dev metrics
                metrics = benchmark.evaluate(preds, qrels)
                logger.info("dev metrics: %s", format_metrics_string(metrics))
                for metric_str in ["AP", "P@20", "NDCG@20"]:
                    metric = convert_metric(metric_str)
                    summary_writer.add_scalar(metric_str, metrics[metric], niter)

                # write best dev weights to file
                if metrics[metric] > dev_best_metric:
                    dev_best_metric = metrics[metric]
                    logger.info("new best dev metric: %0.4f", dev_best_metric)
                    reranker.save_weights(dev_best_weight_fn, self.optimizer)
                    self.write_to_metric_file(metric_fn, metrics)

            # write train_loss to file
            self.write_to_loss_file(loss_fn, train_loss)

            summary_writer.add_scalar("training_loss", iter_loss_tensor.item(), niter)
            reranker.add_summary(summary_writer, niter)
            summary_writer.flush()
        logger.info("training loss: %s", train_loss)
        logger.info("Training took {}".format(time.time() - train_start_time))
        summary_writer.close()

        # TODO should we write a /done so that training can be skipped if possible when fastforward=False? or in Task?

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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # save to pred_fn

        reranker.model.eval()

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
