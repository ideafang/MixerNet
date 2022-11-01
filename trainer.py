import torch
import time
import logging
import copy
import pickle
from newmodel import NewModel

from torch.optim import Adam, Adagrad, SGD
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

class BaseModel(object):
    def __init__(self, data_loader, args, max_len, n_links, valid_dataloader=None):
        self.model = NewModel(args, max_len, n_links)
        self.device = args.device
        self.model.to(self.device)
        self.data_loader = data_loader
        
        self.n_steps = args.max_steps
        self.time_tot = 0
        self.step_tot = 0
        self.args = args

        self.result_file = f".{args.model_dir.split('.')[1]}_result.pt"
        self.valid_dataloader = valid_dataloader

        self.logger = logging.getLogger(args.mode)

        # 选择优化器
        if self.args.optim.lower() == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr = self.args.lr)
        elif self.args.optim.lower() == 'adagrad':
            self.optimizer = Adagrad(self.model.parameters(), lr = self.args.lr)
        else:
            self.optimizer = SGD(self.model.parameters(), lr = self.args.lr)

        # self.scheduler = ExponentialLR(self.optimizer, self.args.decay_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.args.decay_rate, patience=2)
        
        # loss 也是 mse 指标
        self.loss = torch.nn.MSELoss(reduction='mean')
        # mae 指标
        self.mae = torch.nn.L1Loss()

        # AMP On
        self.scaler = GradScaler()
        
        # best_result = 0
        # bad_counts = 0
        # best_model = None
    
    def _train_step(self, bw, tr, p_lidx, mask, y):
        self.model.train()
        start_time = time.time()

        bw, tr, p_lidx, mask = bw.to(self.device), tr.to(self.device), p_lidx.to(self.device), mask.to(self.device)
        y = y.to(self.device).view(-1)

        self.model.zero_grad()
        with autocast():
            y_hat = self.model(bw, tr, p_lidx, mask)
            loss = self.loss(y_hat, y)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        step_loss = loss.data.cpu().numpy()
        self.time_tot += time.time() - start_time
        return step_loss

    # 皮尔森系数计算
    def pearsonr(self, y_pred, y_true):
        y_pred, y_true = y_pred.view(-1), y_true.view(-1)
        centered_pred = y_pred - torch.mean(y_pred)
        centered_true = y_true - torch.mean(y_true)
        covariance = torch.sum(centered_pred * centered_true)
        bessel_corrected_covariance = covariance / (y_pred.size(0) - 1)
        std_pred = torch.std(y_pred, dim=0)
        std_true = torch.std(y_true, dim=0)
        corr = bessel_corrected_covariance / (std_pred * std_true)
        return corr

    def _test(self, dataloader, save_result):
        self.model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for bw, tr, p_lidx, mask, y in dataloader:
                bw, tr, p_lidx, mask = bw.to(self.device), tr.to(self.device), p_lidx.to(self.device), mask.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(bw, tr, p_lidx, mask)
                y_pred.append(y_hat.view(-1).to('cpu'))
                y_true.append(y.view(-1).to('cpu'))
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
            mse = self.loss(y_pred, y_true)
            mae = self.mae(y_pred, y_true)
            rho = self.pearsonr(y_pred, y_true)
            mre = torch.mean(torch.abs(y_true - y_pred) / torch.abs(y_true))
            # 检查是否保存测试结果
            if save_result:
                self.logger.info(f"# saving test results in {self.result_file}")
                with open(self.result_file, 'wb') as f:
                    pickle.dump((y_pred, y_true), f)
        return mse, mae, rho, mre

    def train(self):
        loss = 0
        lr = self.optimizer.param_groups[0]['lr']
        best_model = None
        best_result = 1
        bad_count = 0
        best_step = 0
        training = True
        while training:
            for bw, tr, p_lidx, mask, y in self.data_loader:
                loss += self._train_step(bw, tr, p_lidx, mask, y)
                self.step_tot += 1
                # # 检查是否更新学习率
                # if self.step_tot % self.args.step_update == 0:
                #     self.logger.info("# update learning rate ...")
                #     self.scheduler.step()
                    
                # 检查是否记录进度及更新lr
                if self.step_tot % self.args.log_step == 0:
                    loss /= self.args.log_step
                    log = "Train Step: %d | Loss: %.5f | Time: %.4f" % (self.step_tot, loss, self.time_tot)
                    self.logger.info(log)
                    loss = 0
                
                # 检查是否需要验证
                if self.step_tot % self.args.step_per_test == 0:
                    mse, mae, rho, mre  = self._test(self.valid_dataloader, save_result=False)
                    # 用MRE作为验证指标
                    if mre < best_result:
                        best_result = mre
                        best_step = self.step_tot
                        best_model = copy.deepcopy(self.model.state_dict())
                        bad_count = 0
                    else:
                        bad_count += 1
                    # 记录验证log
                    log = "Eval Step: %d | MSE: %.5f | MAE: %.5f | RHO: %.5f | MRE: %.5f" % (self.step_tot, mse, mae, rho, mre)
                    self.logger.info('='*10 + f"Evaluation in Step {self.step_tot}" + '='*10)
                    self.logger.info(log)
                    self.logger.info('=' * 20)
                    # 判断是否更新learning rate
                    self.scheduler.step(mse)
                    lr_now = self.optimizer.param_groups[0]['lr']
                    if not lr_now == lr:
                        lr = lr_now
                        self.logger.info("# Step: %d: reducing learning rate to %.2e." % (self.step_tot, lr))
                    # 判断是否提前停止训练
                    if bad_count == self.args.earlystopping:
                        log = "Early stopping at step %d" %(self.step_tot)
                        self.logger.info('=' * 20)
                        self.logger.info(log)
                        self.logger.info('=' * 20)
                        training = False
                        break
                
                # 检查是否达到最大step
                if self.step_tot >= self.n_steps:
                    training = False
                    break
        
        # 训练结束
        if best_model == None:
            best_model = self.model.state_dict()
            best_step = self.step_tot
        # 保存最佳模型
        torch.save(best_model, self.args.model_dir)
        log = "Step: %d | Time: %.2f" % (self.step_tot, self.time_tot)
        self.logger.info('=' * 10 + "Training End" + '=' * 10)
        self.logger.info(log)
        self.logger.info('=' * 20)
        self.model.load_state_dict(best_model)
        mse, mae, rho, mre  = self._test(self.valid_dataloader, save_result=True)
        log = "Test Step: %d | MSE: %.5f | MAE: %.5f | RHO: %.5f | MRE: %.5f" % (best_step, mse, mae, rho, mre)
        self.logger.info('='*10 + f"Test in Step {self.step_tot}" + '='*10)
        self.logger.info(log)
        self.logger.info('=' * 20)