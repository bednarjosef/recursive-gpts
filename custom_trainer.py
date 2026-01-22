from __future__ import annotations

import torch
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from ema_pytorch import EMA
from custom_trm import TinyRecursiveModel
from adam_atan2_pytorch import MuonAdamAtan2
from x_transformers import Encoder, Decoder


def evaluate(dataloader, model, max_recurrent_steps, decode, device):
    model.eval()
    print(f'Evaluating...')
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        preds, exit_steps = model.predict(
            x,
            max_deep_refinement_steps = max_recurrent_steps, 
            halt_prob_thres = 0.5
        )
        
        if shown_examples < 5:
            for i in range(x.size(0)):
                if shown_examples >= 5: break
                
                # Decode
                prompt_ids = x[i]
                prompt_ids = prompt_ids[prompt_ids]  #  != pad_id
                prompt_str = decode(prompt_ids.tolist())
                
                truth_ids = y[i]
                truth_str = decode(truth_ids.tolist())
                
                pred_ids = preds[i]
                pred_str = decode(pred_ids.tolist())
                
                print(f'Example {i+1}:')
                print(f'Q: {prompt_str}')
                print(f'Truth: {truth_str}')
                print(f'Pred: {pred_str}\n')
                
                # status = "✅" if row_match[i] else "❌"
                steps = exit_steps[i].item()
                
                print(f"Prob: {prompt_str} | Truth: {truth_str} | Pred: {pred_str} | Steps: {steps} x")
                shown_examples += 1

    model.train()



def exists(v):
    return v is not None

def range_from_one(n):
    return range(1, n + 1)

def is_empty(t):
    return t.numel() == 0

class Trainer(Module):
    def __init__(
        self,
        model: TinyRecursiveModel | Module,
        dataset: Dataset,
        val_dataset: Dataset,
        optim_klass = AdamW,
        optim: Optimizer | None = None,
        learning_rate = 1e-4,
        muon_learning_rate = 1e-3,
        weight_decay = 1.,
        batch_size = 16,
        epochs = 2,
        halt_prob_thres = 0.5,
        max_recurrent_steps = 12,
        warmup_steps = 2000,
        ema_decay_rate = 0.999,
        switch_ema_every = 10000,           # switch ema https://arxiv.org/abs/2402.09240
        accelerate_kwargs: dict = dict(),
        cpu = False,
        eval_interval = 100,
        decode = None,

    ):
        super().__init__()
        self.eval_interval = eval_interval
        self.decode = decode
        self.val_dataloader = DataLoader(val_dataset, batch_size)

        self.accelerator  = Accelerator(**accelerate_kwargs, cpu = cpu)

        self.batch_size = batch_size
        self.epochs = epochs

        # data

        self.dataset = dataset
        self.dataloader = dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True)

        # optim

        if not exists(optim):

            if isinstance(model.network, (Encoder, Decoder)):
                optim = MuonAdamAtan2(
                    model.network.muon_parameters(),
                    model.parameters(),
                    lr = learning_rate / (batch_size * max_recurrent_steps),
                    muon_lr = muon_learning_rate / (batch_size * max_recurrent_steps),
                    weight_decay = weight_decay
                )
            else:
                optim = optim_klass(
                    model.parameters(),
                    lr = learning_rate / (batch_size * max_recurrent_steps),
                    weight_decay = weight_decay
                )

        self.optim = optim

        # scheduler

        self.scheduler = LambdaLR(self.optim, lambda step: min((step + 1) / warmup_steps, 1.0))

        # model

        self.model = model

        # ema model

        self.ema_model = None

        if self.accelerator.is_main_process:
            self.ema_model = EMA(
                model,
                beta = ema_decay_rate,
                update_model_with_ema_every = switch_ema_every,
                forward_method_names = ('predict',)
            )

            self.ema_model.to(self.accelerator.device)

        # recurrent and act related variables

        self.halt_prob_thres = halt_prob_thres

        self.max_recurrent_steps = max_recurrent_steps

        # prepare maybe distributed

        self.model, self.optim, self.dataloader, self.scheduler = self.accelerator.prepare(self.model, self.optim, self.dataloader, self.scheduler)

    def forward(self):
        global_step = 0
        for epoch in range_from_one(self.epochs):

            for dataset_input, dataset_output in self.dataloader:
                global_step += 1
                
                row_main_loss, row_halt_loss = 0, 0
                outputs, latents = self.model.get_initial()
                # print(f'step {global_step}')

                for recurrent_step in range_from_one(self.max_recurrent_steps):

                    loss, (main_loss, halt_loss), outputs, latents, pred, halt = self.model(dataset_input, outputs, latents, labels = dataset_output)
                    row_main_loss += main_loss.mean().item()
                    row_halt_loss += halt_loss.mean().item()

                    # self.accelerator.print(f'[{epoch} ({recurrent_step} / {self.max_recurrent_steps})] loss: {main_loss.mean().item():.3f} | halt loss: {halt_loss.mean().item():.3f}')

                    self.accelerator.backward(loss)

                    self.optim.step()
                    self.optim.zero_grad()

                    self.scheduler.step()

                    if self.accelerator.is_main_process:
                        self.ema_model.update()

                    # handle halting
                    halt_mask = halt >= self.halt_prob_thres

                    if not halt_mask.any():
                        continue

                    outputs = outputs[~halt_mask]
                    latents = latents[~halt_mask]
                    dataset_input = dataset_input[~halt_mask]
                    dataset_output = dataset_output[~halt_mask]

                    if is_empty(outputs):
                        break
                
                if global_step % self.eval_interval == 0:
                    evaluate(self.val_dataloader, self.model, self.max_recurrent_steps, self.decode, self.accelerator.device)
                
                if global_step == 1 or global_step % 10 == 0:
                    row_main_loss /= self.max_recurrent_steps
                    row_halt_loss /= self.max_recurrent_steps
                    self.accelerator.print(f'epoch: {epoch} | step: {global_step} | loss: {row_main_loss:.3f} | halt loss: {row_halt_loss:.3f}')
            

        self.accelerator.print('training complete')

        if self.accelerator.is_main_process:
            self.ema_model.copy_params_from_ema_to_model()
