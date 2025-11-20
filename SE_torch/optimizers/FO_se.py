import torch
from torch.optim import SGD, Muon, AdamW, LBFGS
from tqdm import tqdm

from SE_torch.optimizers.base_optimizer import SEOptimizer, WeightedSELoss, WeightedSEWithPriorLoss


class SGD_se(SEOptimizer):
    def __init__(self, **kwargs):
        super(SGD_se, self).__init__(**kwargs)
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = int(kwargs.get('max_iter', 1e+6))
        self.lr = kwargs.get('lr', 1e-4)
        self.start_factor = kwargs.get('start_factor', 1e-5)
        self.end_factor = kwargs.get('end_factor', 1.5e-3)
        self.verbose = kwargs.get('verbose', True)
        self.use_prior = kwargs.get('use_prior', False)

    def __call__(self, x0, z, v, slk_bus, h_ac, nb ,norm_H=None):
        T = x0[:nb].clone()
        T[slk_bus[0]] = slk_bus[1]
        V = x0[nb:].clone()
        T.requires_grad_(True)
        V.requires_grad_(True)
        non_slk_bus = (torch.arange(nb) != slk_bus[0]).type(torch.float32)
        slk_vec = torch.zeros_like(non_slk_bus)
        slk_vec[slk_bus[0]] = slk_bus[1]

        R = torch.diag(1. / v)
        if self.use_prior:
            criterion = WeightedSEWithPriorLoss(z, R, slk_bus[0], norm_H=norm_H)
        else:
            criterion = WeightedSELoss(z, R, norm_H=norm_H)
        optimizer = SGD([T, V], lr=self.lr, momentum=.9, nesterov=True)

        scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.start_factor, end_factor=self.end_factor, total_iters=self.max_iter)
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                patience=0,
                                                                factor=1/1.001**2,
                                                                threshold_mode='abs',
                                                                threshold=0,
                                                                eps=1e-12)
        converged = False
        it, delta = 0, torch.inf
        loss_prev = torch.inf
        for it in tqdm(range(int(self.max_iter)), desc=f'Optimizing with SGD', leave=False, colour='green'):
            optimizer.zero_grad()
            T_non_slk = T * non_slk_bus + slk_vec
            z_est = h_ac.estimate(T_non_slk, V)
            if self.use_prior:
                x = torch.concat([T_non_slk, V]).unsqueeze(0)
                loss = criterion(z_est, x)
            else:
                loss = criterion(z_est)
            loss.backward()
            optimizer.step()
            delta = loss_prev - loss.item()
            if 0 <= delta < self.tol:
                converged = True
                break
            loss_prev = loss.item()
            scheduler1.step(epoch=None)
            scheduler2.step(loss_prev)

            if self.verbose and it % (self.max_iter / 100) == 0:
                print(
                    f'SGD - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}, lr: {scheduler2.get_last_lr()[0]:.5e}')
        T = T.detach() * non_slk_bus + slk_vec
        x = torch.concat([T, V], dim=0)
        if self.verbose and it > 0:
            print(f'SGD - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}')
        return x, T, V, converged, it


class LBFGS_se(SEOptimizer):
    def __init__(self, **kwargs):
        super(LBFGS_se, self).__init__(**kwargs)
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = int(kwargs.get('max_iter', 500))
        self.verbose = kwargs.get('verbose', True)
        self.use_prior = kwargs.get('use_prior', False)
        self.prior_config_path = kwargs.get('prior_config_path')

    def __call__(self, x0, z, v, slk_bus, h_ac, nb ,norm_H=None):
        T = x0[:nb].clone()
        T[slk_bus[0]] = slk_bus[1]
        V = x0[nb:].clone()
        T.requires_grad_(True)
        V.requires_grad_(True)
        non_slk_bus = (torch.arange(nb) != slk_bus[0]).type(torch.float32)
        slk_vec = torch.zeros_like(non_slk_bus)
        slk_vec[slk_bus[0]] = slk_bus[1]

        R = torch.diag(1.0 / v)
        if self.use_prior:
            criterion = WeightedSEWithPriorLoss(z, R, slk_bus[0], norm_H=norm_H, NF_config_path=self.prior_config_path)
        else:
            criterion = WeightedSELoss(z, R)
        optimizer = LBFGS([T, V], line_search_fn='strong_wolfe')
        lamda = next(criterion.lamda_iter)
        def closure():
            optimizer.zero_grad()
            T_non_slk = T * non_slk_bus + slk_vec
            z_est = h_ac.estimate(T_non_slk, V)
            if self.use_prior:
                x = torch.concat([T_non_slk, V])
                loss = criterion(z_est, x, lamda)
            else:
                loss = criterion(z_est)
            loss.backward()
            return loss

        converged = False
        it, delta = 0, torch.inf
        loss_prev = torch.inf
        for it in tqdm(range(int(self.max_iter)), desc=f'Optimizing with LBFGS', leave=False, colour='green'):
            optimizer.step(closure)
            with torch.no_grad():
                T_non_slk = T * non_slk_bus + slk_vec
                z_est = h_ac.estimate(T_non_slk, V)
                if self.use_prior:
                    x = torch.concat([T_non_slk, V])
                    loss = criterion(z_est, x, lamda)
                else:
                    loss = criterion(z_est)
            delta = loss_prev - loss.item()
            if 0 <= delta < self.tol:
                converged = True
                break
            loss_prev = loss.item()
            lamda = next(criterion.lamda_iter, 1e-2)
            if self.verbose and it % (self.max_iter / 100) == 0:
                print(f'LBFGS - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}')
        T = T.detach() * non_slk_bus + slk_vec
        x = torch.concat([T, V], dim=0)
        if self.verbose and it > 0:
            print(f'LBFGS - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}')
        return x, T, V, converged, it


class Muon_se(SEOptimizer):
    def __init__(self, **kwargs):
        super(Muon_se, self).__init__(**kwargs)
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = int(kwargs.get('max_iter', 1e+6))
        self.lr = kwargs.get('lr', 1e-3)
        self.patience = kwargs.get('patience', 1)
        self.factor = kwargs.get('factor', .99)
        self.threshold = kwargs.get('threshold', 0)
        self.eps = kwargs.get('eps', 1e-12)
        self.verbose = kwargs.get('verbose', True)
        self.use_prior = kwargs.get('use_prior', False)

    def __call__(self, x0, z, v, slk_bus, h_ac, nb ,norm_H=None):
        T = x0[:nb].clone().reshape(-1, 1)
        T[slk_bus[0]] = slk_bus[1]
        V = x0[nb:].clone().reshape(-1, 1)
        T.requires_grad_(True)
        V.requires_grad_(True)
        non_slk_bus = (torch.arange(nb) != slk_bus[0]).type(torch.float32).reshape(-1, 1)
        slk_vec = torch.zeros_like(non_slk_bus).reshape(-1, 1)
        slk_vec[slk_bus[0]] = slk_bus[1]

        R = torch.diag(1.0 / v)
        if self.use_prior:
            criterion = WeightedSEWithPriorLoss(z, R, slk_bus[0], norm_H=norm_H)
        else:
            criterion = WeightedSELoss(z, R, norm_H=norm_H)
        optimizer = Muon([T, V], lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=self.patience,
                                                               factor=self.factor,
                                                               threshold=self.threshold,
                                                               eps=self.eps)
        converged = False
        it, delta = 0, torch.inf
        loss_prev = torch.inf
        for it in tqdm(range(int(self.max_iter)), desc=f'Optimizing with Muon', leave=False, colour='green'):
            optimizer.zero_grad()
            T_non_slk = T * non_slk_bus + slk_vec
            z_est = h_ac.estimate(T_non_slk, V)
            if self.use_prior:
                x = torch.concat([T_non_slk, V]).unsqueeze(0)
                loss = criterion(z_est, x)
            else:
                loss = criterion(z_est)
            loss.backward()
            optimizer.step()
            delta = loss_prev - loss.item()
            if 0 <= delta < self.tol:
                converged = True
                break
            loss_prev = loss.item()
            scheduler.step(loss_prev)

            if self.verbose and it % (self.max_iter / 100) == 0:
                print(
                    f'Muon - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.5e}')
        T = T.detach() * non_slk_bus + slk_vec
        T = T.flatten()
        V = V.flatten()
        x = torch.concat([T, V], dim=0)
        if self.verbose and it > 0:
            print(f'Muon - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.5e}')
        return x, T, V, converged, it


class AdamW_se(SEOptimizer):
    def __init__(self, **kwargs):
        super(AdamW_se, self).__init__(**kwargs)
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = int(kwargs.get('max_iter', 1e+6))
        self.lr = kwargs.get('lr', 1e-3)
        self.patience = kwargs.get('patience', 1)
        self.factor = kwargs.get('factor', .99)
        self.threshold = kwargs.get('threshold', 0)
        self.eps = kwargs.get('eps', 1e-12)
        self.verbose = kwargs.get('verbose', True)
        self.use_prior = kwargs.get('use_prior', False)

    def __call__(self, x0, z, v, slk_bus, h_ac, nb ,norm_H=None):
        T = x0[:nb].clone()
        T[slk_bus[0]] = slk_bus[1]
        V = x0[nb:].clone()
        T.requires_grad_(True)
        V.requires_grad_(True)
        non_slk_bus = (torch.arange(nb) != slk_bus[0]).type(torch.float32)
        slk_vec = torch.zeros_like(non_slk_bus)
        slk_vec[slk_bus[0]] = slk_bus[1]

        R = torch.diag(1.0 / v)
        if self.use_prior:
            criterion = WeightedSEWithPriorLoss(z, R, slk_bus[0], norm_H=norm_H)
        else:
            criterion = WeightedSELoss(z, R, norm_H=norm_H)
        optimizer = AdamW([T, V], lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=self.patience,
                                                               factor=self.factor,
                                                               threshold=self.threshold,
                                                               eps=self.eps)
        converged = False
        it, delta = 0, torch.inf
        loss_prev = torch.inf
        for it in tqdm(range(int(self.max_iter)), desc=f'Optimizing with AdamW', leave=False, colour='green'):
            optimizer.zero_grad()
            T_non_slk = T * non_slk_bus + slk_vec
            z_est = h_ac.estimate(T_non_slk, V)
            if self.use_prior:
                x = torch.concat([T_non_slk, V]).unsqueeze(0)
                loss = criterion(z_est, x)
            else:
                loss = criterion(z_est)
            loss.backward()
            optimizer.step()
            delta = loss_prev - loss.item()
            if 0 <= delta < self.tol:
                converged = True
                break
            loss_prev = loss.item()
            scheduler.step(loss_prev)

            if self.verbose and it % (self.max_iter / 100) == 0:
                print(
                    f'AdamW - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.5e}')
        T = T.detach() * non_slk_bus + slk_vec
        x = torch.concat([T, V], dim=0)
        if self.verbose and it > 0:
            print(f'AdamW - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.5e}')
        return x, T, V, converged, it

