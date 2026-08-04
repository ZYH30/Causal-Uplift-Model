import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kanPro import KAN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import optuna
import torch.nn.utils as nn_utils
from . import utilseed

def clean_tensor(tensor):
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
    tensor = torch.clamp(tensor, min=-1e6, max=1e6)
    return tensor

def modelTrain(model, dataset, lr, steps, loss_fn, device, l2_lambda=1e-4, verbose=True, weights=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for step in range(steps):
        pred = model(dataset['train_input'])

        loss = loss_fn(pred, dataset['train_label'], weights=weights)
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param.pow(2))
        loss += l2_lambda * l2_reg

        optimizer.zero_grad()
        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0 and verbose:
            print(f"Step {step}/{steps}: Loss = {loss.item():.6f}")

    with torch.no_grad():
        for param in model.parameters():
            param.data = torch.clamp(param.data, min=-1e3, max=1e3)
            param.data = torch.nan_to_num(param.data, nan=0.0)

        test_pred = model(dataset['test_input'])
        testloss = loss_fn(test_pred, dataset['test_label'], is_train=False)

    return model, testloss.cpu().numpy()

def kan_fit(X, y, y_type, steps=64, grid=4, k=4, prune_rate=0.008, refine_times=5,
            lamb=1e-5, lamb_l1=1e-4, lamb_entropy=1e-4, lr=1e-2, batch_size=-1,
            hidden_dim=4, opt=False, verbose=False, weights=None, random_state=2025,
            method='random', residual_type='default'):

    utilseed.set_seed(random_state)

    def CE_loss(x, y, is_train=True, weights=None, is_reduction=True):
        if is_train and (weights is not None):
            raw_loss = nn.CrossEntropyLoss(reduction='none')(x, y)
            if is_reduction:
                weighted_loss = dataset['train_weight'] * raw_loss
                final_loss = weighted_loss.sum()
            else:
                return raw_loss.cpu().numpy()
        else:
            if is_reduction:
                final_loss = nn.CrossEntropyLoss()(x, y)
            else:
                return nn.CrossEntropyLoss(reduction='none')(x, y).cpu().numpy()
        return final_loss

    def BCE_loss(x, y, is_train=True, weights=None, is_reduction=True):
        if is_train and (weights is not None):
            raw_loss = nn.BCEWithLogitsLoss(reduction='none')(x.squeeze(), y.float())
            if is_reduction:
                weighted_loss = dataset['train_weight'] * raw_loss
                final_loss = weighted_loss.sum()
            else:
                return raw_loss.cpu().numpy()
        else:
            if is_reduction:
                final_loss = nn.BCEWithLogitsLoss()(x.squeeze(), y.float())
            else:
                return nn.BCEWithLogitsLoss(reduction='none')(x.squeeze(), y.float()).cpu().numpy()
        return final_loss

    def MSE_loss(x, y, is_train=True, weights=None):
        if is_train and (weights is not None):
            final_loss = (dataset['train_weight'] * (x.squeeze() - y)**2).sum()
        else:
            final_loss = nn.MSELoss()(x.squeeze(), y)
        return final_loss

    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device.type.upper()}")

    X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X,
                          dtype=torch.float32).to(device)
    X_tensor = clean_tensor(X_tensor)

    if y_type == 'discrete':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        if len(le.classes_) > 2:
            loss_fn = CE_loss
            y_tensor = torch.tensor(y_encoded, dtype=torch.int64).to(device)
        else:
            loss_fn = BCE_loss
            y_tensor = torch.tensor(y_encoded, dtype=torch.float32).to(device)
    else:
        y_encoded = y
        loss_fn = MSE_loss
        y_tensor = torch.tensor(y_encoded, dtype=torch.float32).to(device)

    X_np = X_tensor.cpu().numpy()
    y_np = y_tensor.cpu().numpy()

    if method == 'quantile':
        train_sample = y_np <= np.quantile(y_np, 0.8)
        val_sample = y_np > np.quantile(y_np, 0.8)

        X_train = X_np[train_sample]
        X_test = X_np[val_sample]
        y_train = y_np[train_sample]
        y_test = y_np[val_sample]

        if weights is not None:
            w_train = weights[train_sample]
            w_test = weights[val_sample]
        else:
            w_train = np.ones(len(X_train))
            w_test = np.ones(len(X_test))
    else:
        if weights is not None:
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X_np, y_np, weights, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, test_size=0.2, random_state=42
            )
            w_train = np.ones(len(X_train))
            w_test = np.ones(len(X_test))

    if y_type == 'discrete':
        dataset = {
            'train_input': torch.tensor(X_train, dtype=torch.float32).to(device),
            'train_label': torch.tensor(y_train, dtype=torch.int64).to(device),
            'train_weight': torch.tensor(w_train, dtype=torch.float32).to(device),
            'test_input': torch.tensor(X_test, dtype=torch.float32).to(device),
            'test_label': torch.tensor(y_test, dtype=torch.int64).to(device),
            'test_weight': torch.tensor(w_test, dtype=torch.float32).to(device)
        }
    else:
        dataset = {
            'train_input': torch.tensor(X_train, dtype=torch.float32).to(device),
            'train_label': torch.tensor(y_train, dtype=torch.float32).to(device),
            'train_weight': torch.tensor(w_train, dtype=torch.float32).to(device),
            'test_input': torch.tensor(X_test, dtype=torch.float32).to(device),
            'test_label': torch.tensor(y_test, dtype=torch.float32).to(device),
            'test_weight': torch.tensor(w_test, dtype=torch.float32).to(device)
        }

    input_dim = X.shape[1]
    output_dim = len(le.classes_) if (y_type == 'discrete' and len(le.classes_) > 2) else 1

    model = KAN(width=[input_dim, hidden_dim, output_dim],
                grid=grid, k=k, device=device,
                seed=random_state if random_state is not None else 2025)

    model, testloss = modelTrain(model, dataset, lr, steps, loss_fn, device,
                               l2_lambda=lamb, verbose=verbose, weights=weights)

    try:
        modelResult = model.prune(node_th=prune_rate, edge_th=prune_rate * 0.5)
        modelResult, testloss = modelTrain(modelResult, dataset, lr, steps, loss_fn, device,
                                    l2_lambda=lamb, verbose=verbose, weights=weights)
    except Exception as e:
        print(f"Model pruning failed: {str(e)}")
        modelResult = model

    if opt:
        return testloss
    else:
        with torch.no_grad():
            X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X,
                                  dtype=torch.float32).to(device)
            pred = modelResult(X_tensor)
            pred_np = pred.cpu().numpy()

        def calculate_residual(y_true, pred_values, pred_tensor, residual_type, num_classes):
            if y_type == 'continuous':
                return y_true - pred_values.flatten()

            elif y_type == 'discrete':
                if num_classes == 1:
                    if residual_type == 'prob' or residual_type == 'default':
                        pred_prob = torch.sigmoid(pred_tensor).cpu().numpy().flatten()
                        return y_true - pred_prob
                    elif residual_type == 'logit':
                        logit = pred_tensor.cpu().numpy().flatten()
                        epsilon = 1e-7
                        y_logit = np.log(y_true + epsilon) - np.log(1 - y_true + epsilon)
                        return y_logit - logit
                    elif residual_type == 'cross_entropy':
                        pred_prob = torch.sigmoid(pred_tensor).cpu().numpy().flatten()
                        epsilon = 1e-7
                        pred_prob = np.clip(pred_prob, epsilon, 1 - epsilon)
                        return -(y_true * np.log(pred_prob) + (1 - y_true) * np.log(1 - pred_prob))
                else:
                    if residual_type == 'weighted_prob' or residual_type == 'default':
                        y_onehot = F.one_hot(torch.tensor(y_true, dtype=torch.int64),
                                           num_classes=num_classes)
                        pred_prob = F.softmax(pred_tensor, dim=1).cpu()
                        prob_residual = pred_prob - y_onehot.float()
                        return torch.norm(prob_residual, dim=1).numpy()
                    elif residual_type == 'cross_entropy':
                        return nn.CrossEntropyLoss(reduction='none')(pred_tensor,
                                                                   torch.tensor(y_true, dtype=torch.int64).to(pred_tensor.device)).cpu().numpy()
                    elif residual_type == 'normalized_cross_entropy':
                        cross_entropy = nn.CrossEntropyLoss(reduction='none')(
                            pred_tensor, torch.tensor(y_true, dtype=torch.int64).to(pred_tensor.device))
                        max_entropy = -np.log(1.0/num_classes)
                        return (cross_entropy / max_entropy).cpu().numpy()
            return None

        if y_type == 'discrete':
            if output_dim > 1:
                y_pred = pred_np.argmax(1)
                metric_score = accuracy_score(y_encoded, y_pred)
                residual = calculate_residual(y_encoded, pred_np, pred, residual_type, output_dim)
            else:
                y_pred = (pred_np >= 0.5).astype(int)
                metric_score = roc_auc_score(y_encoded, pred_np.flatten()) if len(le.classes_) == 2 else accuracy_score(y_encoded, y_pred)
                residual = calculate_residual(y_encoded, pred_np, pred, residual_type, 1)
        else:
            metric_score = mean_squared_error(y_encoded, pred_np)
            residual = calculate_residual(y_encoded, pred_np, pred, residual_type, output_dim)

        return modelResult, 1, residual, metric_score

def kan_optuna(X, y, y_type, n_trials=12, weights=None, lamb=1e-4, random_state=2025, residual_type='default', is_optM=False):

    utilseed.set_seed(random_state)
    print('n_trials:', n_trials)
    print('lamb:', lamb)
    steps = 32
    dim = X.shape[1]

    if is_optM:
        if dim <= 2:
            dim_lower = 2
            dim_upper = 6
        elif dim >= 15:
            dim_lower = 8
            dim_upper = 12
        else:
            dim_lower = max(2, round(0.5 * dim))
            dim_upper = int(1.5 * dim)

        print('dim_lower: ', dim_lower)
        print('dim_upper: ', dim_upper)

        def objectiveFun(trial):
            params = {
                'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
                'hidden_dim': trial.suggest_int('hidden_dim', dim_lower, dim_upper),
                'grid': trial.suggest_int('grid', 5, 20),
                'prune_rate': trial.suggest_float('prune_rate', 0.001, 0.01),
                'k': trial.suggest_int('k', 5, 20)
            }

            loss = kan_fit(X, y, y_type, steps=steps, opt=True, weights=weights,
                         lamb=lamb, verbose=False, random_state=random_state, **params
                         )
            return loss

        direction = 'minimize'
        sampler = optuna.samplers.TPESampler(seed=random_state, n_startup_trials=64)
        study = optuna.create_study(direction=direction, sampler=sampler)

        try:
            study.optimize(objectiveFun, n_trials=n_trials, catch=(ValueError, RuntimeError))
        except Exception as e:
            print(f"Optuna optimization exception: {str(e)}")

        valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not valid_trials:
            raise RuntimeError("All trials failed, please check data or parameter ranges")

        best_params = study.best_params
        print(f"Best parameters: {best_params}")
        print(f"Best objective value: {study.best_value}")
    else:
        best_params = {
            'lr': 0.01,
            'hidden_dim': int(2.5 * dim),
            'grid': 10,
            'prune_rate': 0.002,
            'k': 10
        }
        print(f"Using fixed parameters (Skipping Optuna optimization): {best_params}")

    model, _, residual, loss = kan_fit(X, y, y_type, steps=steps, opt=False,
                                     weights=weights, lamb=lamb, verbose=True,
                                     residual_type=residual_type,
                                     random_state=random_state, **best_params
                                     )

    return model, best_params, residual, loss
