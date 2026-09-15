from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import (
    Autoformer,
    CycleNet,
    DLinear,
    GTR,
    GTRDLinear,
    GTRiTransformer,
    GTRNTE,
    GTRPatchTST,
    Informer,
    Linear,
    NLinear,
    PatchTST,
    SegRNN,
    TimeXer,
    Transformer,
    iTransformer,
)
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.perturbation import (
    PERTURBATION_NONE,
    apply_input_perturbation,
    perturbation_tag,
)
from utils.boundary_fix import apply_boundary_fix
from utils.boundary_reconstruction import (
    BoundaryReconstructor,
    apply_filtered_boundary_reconstruction,
)
from utils.experiment import numeric_tag

import json
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SegRNN': SegRNN,
            'CycleNet': CycleNet,
            'iTransformer': iTransformer,
            'TimeXer': TimeXer,
            'GTR': GTR,
            'GTRNTE': GTRNTE,
            'GTRDLinear': GTRDLinear,
            'GTRPatchTST': GTRPatchTST,
            'GTRiTransformer': GTRiTransformer
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def _forward_test_batch(
        self, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle
    ):
        """Run one inference batch using the repository's model dispatch rules."""

        if any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
            return self.model(batch_x, batch_cycle)
        if any(
            substr in self.args.model
            for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}
        ):
            return self.model(batch_x)
        if self.args.output_attention:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    @staticmethod
    def _primary_metrics(predictions, targets):
        mae, mse, _, _, _, _, _ = metric(predictions, targets)
        return {'mse': float(mse), 'mae': float(mae)}

    @staticmethod
    def _metric_degradation(clean_metrics, perturbed_metrics):
        absolute = {}
        relative_percent = {}
        for name, clean_value in clean_metrics.items():
            delta = perturbed_metrics[name] - clean_value
            absolute[name] = float(delta)
            relative_percent[name] = (
                float(delta / clean_value * 100.0)
                if clean_value != 0
                else None
            )
        return absolute, relative_percent

    @staticmethod
    def _metric_improvement(clean_metrics, fixed_metrics):
        absolute = {}
        relative_percent = {}
        for name, clean_value in clean_metrics.items():
            delta = clean_value - fixed_metrics[name]
            absolute[name] = float(delta)
            relative_percent[name] = (
                float(delta / clean_value * 100.0)
                if clean_value != 0
                else None
            )
        return absolute, relative_percent

    def _boundary_reconstructor_path(self, setting):
        return os.path.join(
            self.args.checkpoints,
            setting,
            'boundary_reconstructor.pth',
        )

    def _new_boundary_reconstructor(self):
        return BoundaryReconstructor(
            seq_len=int(self.args.seq_len),
            channels=int(self.args.enc_in),
            hidden_dim=int(self.args.boundary_hidden_dim),
        ).to(self.device)

    def train_boundary_reconstructor(self, setting):
        """Train the auxiliary prefix-to-boundary model independently of GTR."""

        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        reconstructor = self._new_boundary_reconstructor()
        optimizer = optim.Adam(
            reconstructor.parameters(),
            lr=float(self.args.boundary_learning_rate),
        )
        criterion = nn.SmoothL1Loss()
        checkpoint_path = self._boundary_reconstructor_path(setting)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        best_validation_loss = np.inf
        stale_epochs = 0

        for epoch in range(int(self.args.boundary_epochs)):
            reconstructor.train()
            train_losses = []
            for batch in train_loader:
                batch_x = batch[0].float().to(self.device)
                optimizer.zero_grad()
                predicted_last = reconstructor(batch_x)
                loss = criterion(predicted_last, batch_x[:, -1, :])
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            reconstructor.eval()
            validation_losses = []
            with torch.no_grad():
                for batch in vali_loader:
                    batch_x = batch[0].float().to(self.device)
                    predicted_last = reconstructor(batch_x)
                    validation_losses.append(
                        float(
                            criterion(
                                predicted_last, batch_x[:, -1, :]
                            ).item()
                        )
                    )

            train_loss = float(np.mean(train_losses))
            validation_loss = float(np.mean(validation_losses))
            print(
                'Boundary reconstructor epoch {} | train loss:{:.7f} '
                'vali loss:{:.7f}'.format(
                    epoch + 1, train_loss, validation_loss
                )
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                stale_epochs = 0
                torch.save(reconstructor.state_dict(), checkpoint_path)
            else:
                stale_epochs += 1
                if stale_epochs >= int(self.args.boundary_patience):
                    print('Boundary reconstructor early stopping')
                    break

        reconstructor.load_state_dict(torch.load(checkpoint_path))
        reconstructor.eval()
        self.boundary_reconstructor = reconstructor
        return reconstructor

    def _load_boundary_reconstructor(self, setting):
        checkpoint_path = self._boundary_reconstructor_path(setting)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                'Boundary reconstructor checkpoint not found: {}'.format(
                    checkpoint_path
                )
            )
        reconstructor = self._new_boundary_reconstructor()
        reconstructor.load_state_dict(torch.load(checkpoint_path))
        reconstructor.eval()
        self.boundary_reconstructor = reconstructor
        return reconstructor

    def test(self, setting, test=0):
        _, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(
                    os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
                )
            )

        perturb_type = getattr(self.args, 'perturb_type', PERTURBATION_NONE)
        perturb_enabled = perturb_type != PERTURBATION_NONE
        perturb_ratio = float(getattr(self.args, 'perturb_ratio', 3.0))
        perturb_seed = int(getattr(self.args, 'perturb_seed', 2024))
        perturb_offset = int(getattr(self.args, 'perturb_offset', 1))
        boundary_fix_enabled = bool(getattr(self.args, 'boundary_fix', 0))
        boundary_reconstruct_enabled = bool(
            getattr(self.args, 'boundary_reconstruct', 0)
        )
        boundary_view_enabled = (
            boundary_fix_enabled or boundary_reconstruct_enabled
        )
        boundary_threshold = float(
            getattr(self.args, 'boundary_threshold', 3.0)
        )
        if boundary_reconstruct_enabled:
            reconstructor = getattr(self, 'boundary_reconstructor', None)
            if reconstructor is None:
                reconstructor = self._load_boundary_reconstructor(setting)
            reconstructor.eval()
        perturb_rng = (
            np.random.RandomState(perturb_seed) if perturb_enabled else None
        )

        clean_preds = []
        perturbed_preds = []
        fixed_preds = []
        trues = []
        boundary_replaced_count = 0
        boundary_value_count = 0
        boundary_score_sum = 0.0
        boundary_reconstruction_squared_error = 0.0
        boundary_persistence_squared_error = 0.0
        boundary_error_value_count = 0
        flagged_reconstruction_squared_error = 0.0
        flagged_input_squared_error = 0.0
        flagged_error_value_count = 0
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_cycle,
            ) in enumerate(test_loader):
                if perturb_enabled:
                    # Apply the perturbation before the repository's float32/device
                    # conversion. This mirrors TameR's NumPy-side perturbation and
                    # then sends both clean and perturbed copies through the same
                    # model path.
                    perturbed_batch_x = apply_input_perturbation(
                        batch_x,
                        perturb_type=perturb_type,
                        perturb_ratio=perturb_ratio,
                        rng=perturb_rng,
                        perturb_offset=perturb_offset,
                    ).float().to(self.device)
                if boundary_fix_enabled:
                    fixed_batch_x = apply_boundary_fix(batch_x).float().to(
                        self.device
                    )
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                if boundary_reconstruct_enabled:
                    reconstruction_input = (
                        perturbed_batch_x if perturb_enabled else batch_x
                    )
                    predicted_last = reconstructor(reconstruction_input)
                    fixed_batch_x, replacement_mask, boundary_scores = (
                        apply_filtered_boundary_reconstruction(
                            reconstruction_input,
                            predicted_last,
                            threshold=boundary_threshold,
                        )
                    )
                    true_last = batch_x[:, -1, :]
                    persistence_last = batch_x[:, -2, :]
                    boundary_replaced_count += int(
                        replacement_mask.sum().item()
                    )
                    boundary_value_count += replacement_mask.numel()
                    boundary_score_sum += float(boundary_scores.sum().item())
                    boundary_reconstruction_squared_error += float(
                        ((predicted_last - true_last) ** 2).sum().item()
                    )
                    boundary_persistence_squared_error += float(
                        ((persistence_last - true_last) ** 2).sum().item()
                    )
                    boundary_error_value_count += true_last.numel()
                    if replacement_mask.any():
                        flagged_reconstruction_squared_error += float(
                            (
                                (predicted_last - true_last)[replacement_mask]
                                ** 2
                            ).sum().item()
                        )
                        flagged_input_squared_error += float(
                            (
                                (
                                    reconstruction_input[:, -1, :]
                                    - true_last
                                )[replacement_mask]
                                ** 2
                            ).sum().item()
                        )
                        flagged_error_value_count += int(
                            replacement_mask.sum().item()
                        )

                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]
                ).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                def run_forward(input_batch):
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            return self._forward_test_batch(
                                input_batch,
                                batch_x_mark,
                                dec_inp,
                                batch_y_mark,
                                batch_cycle,
                            )
                    return self._forward_test_batch(
                        input_batch,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                        batch_cycle,
                    )

                clean_outputs = run_forward(batch_x)
                if perturb_enabled:
                    perturbed_outputs = run_forward(perturbed_batch_x)
                if boundary_view_enabled:
                    fixed_outputs = run_forward(fixed_batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                clean_outputs = clean_outputs[
                    :, -self.args.pred_len:, f_dim:
                ]
                target = batch_y[:, -self.args.pred_len:, f_dim:]
                clean_pred = clean_outputs.detach().cpu().numpy()
                true = target.detach().cpu().numpy()

                clean_preds.append(clean_pred)
                trues.append(true)
                if perturb_enabled:
                    perturbed_outputs = perturbed_outputs[
                        :, -self.args.pred_len:, f_dim:
                    ]
                    perturbed_preds.append(
                        perturbed_outputs.detach().cpu().numpy()
                    )
                if boundary_view_enabled:
                    fixed_outputs = fixed_outputs[
                        :, -self.args.pred_len:, f_dim:
                    ]
                    fixed_preds.append(fixed_outputs.detach().cpu().numpy())

                if i % 20 == 0:
                    input_values = batch_x.detach().cpu().numpy()
                    gt = np.concatenate(
                        (input_values[0, :, -1], true[0, :, -1]), axis=0
                    )
                    plotted_prediction = np.concatenate(
                        (input_values[0, :, -1], clean_pred[0, :, -1]), axis=0
                    )
                    visual(
                        gt,
                        plotted_prediction,
                        os.path.join(folder_path, str(i) + '.pdf'),
                    )

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            return None

        clean_preds = np.concatenate(clean_preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        clean_preds = clean_preds.reshape(
            -1, clean_preds.shape[-2], clean_preds.shape[-1]
        )
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        result_path = './results/' + setting + '/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        clean_metrics = self._primary_metrics(clean_preds, trues)
        result = {
            'schema_version': 1,
            'setting': setting,
            'model_id': self.args.model_id,
            'model': self.args.model,
            'dataset': getattr(self.args, 'dataset_name', None)
            or os.path.splitext(os.path.basename(self.args.data_path))[0],
            'data': self.args.data,
            'data_path': self.args.data_path,
            'features': self.args.features,
            'seq_len': int(self.args.seq_len),
            'pred_len': int(self.args.pred_len),
            'cycle': int(self.args.cycle),
            'train_seed': int(self.args.random_seed),
            'clean': clean_metrics,
        }
        if self.args.model == 'GTRNTE':
            result['plugin'] = {
                'name': 'NTE',
                'parameter_free': True,
                'cutoff_ratio': float(
                    getattr(self.args, 'nte_cutoff_ratio', 0.1)
                ),
                'alpha': float(getattr(self.args, 'nte_alpha', 1.0)),
                'gamma_max': float(
                    getattr(self.args, 'nte_gamma_max', 20.0)
                ),
                'guard_sigma': float(
                    getattr(self.args, 'nte_guard_sigma', 3.0)
                ),
            }

        print(
            'clean mse:{}, mae:{}'.format(
                clean_metrics['mse'], clean_metrics['mae']
            )
        )

        if perturb_enabled:
            perturbed_preds = np.concatenate(perturbed_preds, axis=0)
            perturbed_preds = perturbed_preds.reshape(
                -1, perturbed_preds.shape[-2], perturbed_preds.shape[-1]
            )
            perturbed_metrics = self._primary_metrics(perturbed_preds, trues)
            absolute, relative_percent = self._metric_degradation(
                clean_metrics, perturbed_metrics
            )
            result.update(
                {
                    'perturbation': {
                        'type': perturb_type,
                        'ratio': perturb_ratio,
                        'seed': perturb_seed,
                        'offset': (
                            perturb_offset if perturb_type == 'point' else 1
                        ),
                        'definition': (
                            'x[:, -offset, :] += N(0, 1) * '
                            'std(x, axis=time, ddof=0) * ratio'
                        ),
                    },
                    'perturbed': perturbed_metrics,
                    'degradation_absolute': absolute,
                    'degradation_percent': relative_percent,
                }
            )
            print(
                '{} perturbation (offset={}, ratio={}, seed={}) '
                'mse:{}, mae:{}'.format(
                    perturb_type,
                    perturb_offset if perturb_type == 'point' else 1,
                    perturb_ratio,
                    perturb_seed,
                    perturbed_metrics['mse'],
                    perturbed_metrics['mae'],
                )
            )
            print(
                'degradation mse:{}, mae:{}'.format(
                    (
                        '{:.2f}%'.format(relative_percent['mse'])
                        if relative_percent['mse'] is not None
                        else 'undefined'
                    ),
                    (
                        '{:.2f}%'.format(relative_percent['mae'])
                        if relative_percent['mae'] is not None
                        else 'undefined'
                    ),
                )
            )
            result_file = perturbation_tag(
                perturb_type,
                perturb_ratio,
                perturb_seed,
                perturb_offset=perturb_offset,
            ) + '.json'
        if boundary_reconstruct_enabled:
            fixed_preds = np.concatenate(fixed_preds, axis=0)
            fixed_preds = fixed_preds.reshape(
                -1, fixed_preds.shape[-2], fixed_preds.shape[-1]
            )
            fixed_metrics = self._primary_metrics(fixed_preds, trues)
            reference_metrics = (
                perturbed_metrics if perturb_enabled else clean_metrics
            )
            improvement_absolute, improvement_percent = (
                self._metric_improvement(reference_metrics, fixed_metrics)
            )
            replacement_rate = (
                boundary_replaced_count / boundary_value_count
            )
            flagged_reconstruction_mse = (
                flagged_reconstruction_squared_error
                / flagged_error_value_count
                if flagged_error_value_count
                else None
            )
            flagged_input_mse = (
                flagged_input_squared_error / flagged_error_value_count
                if flagged_error_value_count
                else None
            )
            result.update(
                {
                    'boundary_reconstruction': {
                        'enabled': True,
                        'method': 'context_mlp_delta',
                        'causal': True,
                        'input_prefix_length': int(self.args.seq_len) - 1,
                        'hidden_dim': int(self.args.boundary_hidden_dim),
                        'training_loss': 'smooth_l1',
                        'max_training_epochs': int(
                            getattr(self.args, 'boundary_epochs', 20)
                        ),
                        'training_patience': int(
                            getattr(self.args, 'boundary_patience', 5)
                        ),
                        'learning_rate': float(
                            getattr(
                                self.args,
                                'boundary_learning_rate',
                                0.001,
                            )
                        ),
                        'filter': 'last_increment_mad',
                        'threshold': boundary_threshold,
                        'evaluation_input': (
                            'perturbed' if perturb_enabled else 'clean'
                        ),
                        'replacement_rate': float(replacement_rate),
                        'mean_filter_score': float(
                            boundary_score_sum / boundary_value_count
                        ),
                        'reconstruction_mse': float(
                            boundary_reconstruction_squared_error
                            / boundary_error_value_count
                        ),
                        'persistence_mse': float(
                            boundary_persistence_squared_error
                            / boundary_error_value_count
                        ),
                        'flagged_reconstruction_mse': (
                            float(flagged_reconstruction_mse)
                            if flagged_reconstruction_mse is not None
                            else None
                        ),
                        'flagged_input_mse': (
                            float(flagged_input_mse)
                            if flagged_input_mse is not None
                            else None
                        ),
                    },
                }
            )
            if perturb_enabled:
                residual_absolute, residual_percent = (
                    self._metric_degradation(clean_metrics, fixed_metrics)
                )
                result.update(
                    {
                        'repaired': fixed_metrics,
                        'recovery_absolute': improvement_absolute,
                        'recovery_percent': improvement_percent,
                        'residual_degradation_absolute': residual_absolute,
                        'residual_degradation_percent': residual_percent,
                    }
                )
                print(
                    'filtered boundary-repaired mse:{}, mae:{}'.format(
                        fixed_metrics['mse'], fixed_metrics['mae']
                    )
                )
            else:
                result.update(
                    {
                        'fixed': fixed_metrics,
                        'improvement_absolute': improvement_absolute,
                        'improvement_percent': improvement_percent,
                    }
                )
                print(
                    'filtered boundary-reconstructed mse:{}, mae:{}'.format(
                        fixed_metrics['mse'], fixed_metrics['mae']
                    )
                )
            print(
                'replacement rate:{:.2f}%'.format(replacement_rate * 100.0)
            )
            print(
                'boundary reconstruction mse:{:.6f}, persistence mse:{:.6f}'.format(
                    result['boundary_reconstruction']['reconstruction_mse'],
                    result['boundary_reconstruction']['persistence_mse'],
                )
            )
            if flagged_reconstruction_mse is not None:
                print(
                    'flagged input mse:{:.6f}, reconstructed mse:{:.6f}'.format(
                        flagged_input_mse,
                        flagged_reconstruction_mse,
                    )
                )
            print(
                '{} mse:{}, mae:{}'.format(
                    'recovery' if perturb_enabled else 'improvement',
                    (
                        '{:.2f}%'.format(improvement_percent['mse'])
                        if improvement_percent['mse'] is not None
                        else 'undefined'
                    ),
                    (
                        '{:.2f}%'.format(improvement_percent['mae'])
                        if improvement_percent['mae'] is not None
                        else 'undefined'
                    ),
                )
            )
            if perturb_enabled:
                result_file = '{}_boundary_reconstruct_mad{}.json'.format(
                    perturbation_tag(
                        perturb_type,
                        perturb_ratio,
                        perturb_seed,
                        perturb_offset=perturb_offset,
                    ),
                    numeric_tag(boundary_threshold),
                )
            else:
                result_file = 'boundary_reconstruct_mad{}.json'.format(
                    numeric_tag(boundary_threshold)
                )
        elif boundary_fix_enabled:
            fixed_preds = np.concatenate(fixed_preds, axis=0)
            fixed_preds = fixed_preds.reshape(
                -1, fixed_preds.shape[-2], fixed_preds.shape[-1]
            )
            fixed_metrics = self._primary_metrics(fixed_preds, trues)
            improvement_absolute, improvement_percent = (
                self._metric_improvement(clean_metrics, fixed_metrics)
            )
            result.update(
                {
                    'boundary_fix': {
                        'enabled': True,
                        'method': 'last_value',
                        'definition': 'x[:, -1, :] = x[:, -2, :]',
                    },
                    'fixed': fixed_metrics,
                    'improvement_absolute': improvement_absolute,
                    'improvement_percent': improvement_percent,
                }
            )
            print(
                'boundary-fixed mse:{}, mae:{}'.format(
                    fixed_metrics['mse'], fixed_metrics['mae']
                )
            )
            print(
                'improvement mse:{}, mae:{}'.format(
                    (
                        '{:.2f}%'.format(improvement_percent['mse'])
                        if improvement_percent['mse'] is not None
                        else 'undefined'
                    ),
                    (
                        '{:.2f}%'.format(improvement_percent['mae'])
                        if improvement_percent['mae'] is not None
                        else 'undefined'
                    ),
                )
            )
            result_file = 'boundary_fix_last_value.json'
        elif not perturb_enabled:
            result_file = 'clean_metrics.json'

        with open(
            os.path.join(result_path, result_file), 'w', encoding='utf-8'
        ) as output_file:
            json.dump(result, output_file, indent=2, sort_keys=True)
            output_file.write('\n')

        with open('result.txt', 'a', encoding='utf-8') as output_file:
            output_file.write(setting + '  \n')
            output_file.write(
                'clean mse:{}, mae:{}\n'.format(
                    clean_metrics['mse'], clean_metrics['mae']
                )
            )
            if perturb_enabled:
                output_file.write(
                    '{} ratio:{} seed:{} mse:{}, mae:{}\n'.format(
                        perturb_type,
                        perturb_ratio,
                        perturb_seed,
                        result['perturbed']['mse'],
                        result['perturbed']['mae'],
                    )
                )
            if boundary_reconstruct_enabled and not perturb_enabled:
                output_file.write(
                    'boundary_reconstruct:context_mlp_delta threshold:{} '
                    'mse:{}, mae:{}, mse_improvement:{:.2f}%, '
                    'mae_improvement:{:.2f}%, replacement_rate:{:.2f}%\n'.format(
                        boundary_threshold,
                        result['fixed']['mse'],
                        result['fixed']['mae'],
                        result['improvement_percent']['mse'],
                        result['improvement_percent']['mae'],
                        result['boundary_reconstruction']['replacement_rate']
                        * 100.0,
                    )
                )
            elif boundary_reconstruct_enabled:
                output_file.write(
                    'boundary_repair:context_mlp_delta threshold:{} '
                    'mse:{}, mae:{}, mse_recovery:{:.2f}%, '
                    'mae_recovery:{:.2f}%, replacement_rate:{:.2f}%\n'.format(
                        boundary_threshold,
                        result['repaired']['mse'],
                        result['repaired']['mae'],
                        result['recovery_percent']['mse'],
                        result['recovery_percent']['mae'],
                        result['boundary_reconstruction']['replacement_rate']
                        * 100.0,
                    )
                )
            elif boundary_fix_enabled:
                output_file.write(
                    'boundary_fix:last_value mse:{}, mae:{}, '
                    'mse_improvement:{:.2f}%, mae_improvement:{:.2f}%\n'.format(
                        result['fixed']['mse'],
                        result['fixed']['mae'],
                        result['improvement_percent']['mse'],
                        result['improvement_percent']['mae'],
                    )
                )
            output_file.write('\n')

        return result

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
