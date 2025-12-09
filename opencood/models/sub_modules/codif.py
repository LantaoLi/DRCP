import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.sub_modules.bev_dif_masker import BEVDiffuseMasker, BEVDiffusePerMasker

# ----(1) 简单余弦 beta 日程（可直接换成你工程里的 utils.cosine_beta_schedule）----
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)

# ----(2) 从附件 diffuser.py 抽来的 extract 工具等价实现（保持行为一致）:contentReference[oaicite:2]{index=2}----
def extract(a, t, x_shape):
    """
    a: [T], t: [B] or [B,1], 输出 shape 对齐 x_shape
    """
    if t.dim() == 2:
        t = t.squeeze(1)
    out = a.gather(-1, t)
    return out.reshape(-1, 1, 1, 1).expand(x_shape)

# ----(3) 采样器包装：把任意 denoiser(U-Net) 封进来，提供训练/采样一体接口----
class ConditionedUNetDiffuser(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        in_channels: int,
        num_timesteps: int = 1000,
        sampling_timesteps: int = 50,
        sampler: str = 'ddim',               # 'ddim' | 'dpmsolver' | 'deis'
        objective: str = 'pred_x0',          # 默认预测特征
        solver_type: str = 'midpoint',       # for dpm-solver 2/3 阶
        solver_order: int = 2,               # 1/2/3 阶
        lower_order_final: bool = True,
        ddim_sampling_eta: float = 0.0,
        return_intermediate: bool = False
    ):
        super().__init__()
        assert sampler in ['ddim', 'dpmsolver', 'deis']
        assert objective in ['pred_x0', 'pred_noise']
        self.denoiser = denoiser
        self.in_channels = in_channels
        self.num_timesteps = num_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.sampler = sampler
        self.objective = objective
        self.solver_type = solver_type
        self.solver_order = solver_order
        self.lower_order_final = lower_order_final
        self.ddim_sampling_eta = ddim_sampling_eta
        self.return_intermediate = return_intermediate
        self.masker_1 = BEVDiffusePerMasker(in_channels=256, groups=256)
        self.masker_2 = BEVDiffusePerMasker(in_channels=256)
        # --- 与附件一致：构建 alpha/beta 及其各种组合的 buffer（命名沿用）:contentReference[oaicite:3]{index=3}
        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # ---- DPM/DEIS 需要的 sigma_t / lambda_t（与附件保持一致）:contentReference[oaicite:4]{index=4}
        sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        lambda_t = torch.log(torch.sqrt(alphas_cumprod)) - torch.log(sigma_t)
        self.register_buffer('sigma_t', sigma_t)
        self.register_buffer('lambda_t', lambda_t)

        # posterior（与附件一致）:contentReference[oaicite:5]{index=5}
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # multi-step 存储（与附件一致）:contentReference[oaicite:6]{index=6}
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0

    # ---------------- 基本变换（与附件一致）---------------- :contentReference[oaicite:7]{index=7}
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        ).float()

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        ).float()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return (sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise).float()

    # ----------- 用 denoiser 做一次前向，输出 (x0_pred, noise_pred) -----------
    def model_prediction(self, x_t, cond, ts):
        """
        x_t: [B,C,H,W] 当前时刻样本
        cond: [B,C,H,W] 条件（同形状）
        ts: [B] 或 [1] 的整型时间步
        """
        if ts.dim() == 1:
            ts_in = ts
        else:
            ts_in = ts.view(-1)
        # 你的 ConditionalUNet 默认预测 x0（特征）
        if self.objective == 'pred_x0':
            x0_pred = self.denoiser(x_t, ts_in, cond)
            noise_pred = self.predict_noise_from_start(x_t, ts_in, x0_pred)
        else:
            noise_pred = self.denoiser(x_t, ts_in, cond)
            x0_pred = self.predict_start_from_noise(x_t, ts_in, noise_pred)

        return x0_pred, noise_pred

    # ----------------- 训练/推理统一入口 -----------------
    def forward(self, x_t_or_x0_like, cond, train_flag=True):
        """
        训练：传入 x0_like（干净特征），内部会随机 t 做 q_sample，再监督 MSE。
        推理：传入 x_t（通常从 N(0,I) 采样的起点），按 self.sampler 走采样链。
        为了接口直观：我这里根据 self.training 分支。
        """
        B = x_t_or_x0_like.size(0)

        #was if if self.training
        if train_flag:
            print("training codif")
            # 训练：随机 t，q_sample，监督 objective
            t = torch.randint(0, self.num_timesteps, (B,), device=cond.device).long()
            noise = torch.randn_like(cond)
            x_t = self.q_sample(cond, t, noise=noise) #was self.q_sample(x_t_or_x0_like, t, noise=noise)

            x0_pred, noise_pred = self.model_prediction(x_t, cond, t)

            fuse_w = self.masker_2(cond)
            x0_pred = x0_pred*(1.0-fuse_w) + cond*fuse_w

            if self.objective == 'pred_x0':
                loss = F.l1_loss(x0_pred, x_t_or_x0_like) #was mse_loss(x0_pred, x_t_or_x0_like)
                out = x0_pred
            else:
                loss = F.l1_loss(noise_pred, noise)
                out = noise_pred

            return out, loss, x_t
            #return out, loss, []  # 与附件保持 (x_pred, diffuse_loss, intermediate) 的返回位次一致:contentReference[oaicite:8]{index=8}
        else:
            # 推理：x_t_or_x0_like 视作初始 x_T
            print("evaluating codif")
            t = torch.full((B,), self.num_timesteps-1, device=x_t_or_x0_like.device).long()
            noise = torch.randn_like(x_t_or_x0_like)
            x_t = self.q_sample(x_t_or_x0_like, t, noise=noise)
            if self.sampler == 'ddim':
                x0_pred, pred_noise, inter = self.ddim_sampling(x_t, cond)

                #fuse_w = self.masker_2(cond)
                #x0_pred = x0_pred*(1.0-fuse_w) + cond*fuse_w
                return x0_pred, pred_noise, inter
            elif self.sampler in ['dpmsolver', 'deis']:
                x0_pred, pred_noise, inter = self.dpm_deis_sampling(x_t, cond)

                #fuse_w = self.masker_2(cond)
                #x0_pred = x0_pred*(1.0-fuse_w) + cond*fuse_w
                return x0_pred, pred_noise, inter
            else:
                raise NotImplementedError

    # ----------------- DDIM 采样（来自附件，做了极小改动以适配接口）----------------- :contentReference[oaicite:9]{index=9}
    #@torch.no_grad()
    def ddim_sampling(self, x_t, cond):
        T, S, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        times = torch.linspace(-1, T - 1, steps=S + 1).to(x_t.device).int().tolist()
        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))

        inter = []
        pred_noise = None
        for time, time_next in time_pairs:
            ts = torch.tensor([time], device=x_t.device, dtype=torch.long)
            x0_pred, pred_noise = self.model_prediction(x_t, cond, ts)
            if time_next < 0:
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x_t)
            # 注意：附件里把一步后的 sample 变量也叫 x_t，为一致我仍沿用该命名
            x_t = x0_pred * alpha_next.sqrt() + c * pred_noise + sigma * noise

            fuse_w = self.masker_2(cond)
            x_t = x_t*(1.0-fuse_w) + cond*fuse_w

            if self.return_intermediate:
                inter.append(x_t)
        return x_t, pred_noise, inter

    # ----------------- DPM / DEIS 采样主循环（核心来自附件）----------------- :contentReference[oaicite:10]{index=10}
    #@torch.no_grad()
    def dpm_deis_sampling(self, x_t, cond):
        T, S = self.num_timesteps, self.sampling_timesteps
        times = torch.linspace(-1, T - 1, steps=S + 1).to(x_t.device).int().tolist()
        times = list(reversed(times))
        self.timesteps = times[:-1]
        time_pairs = list(zip(times[:-1], times[1:]))

        inter = []
        pred_noise = None
        self.lower_order_nums = 0  # 每次采样重置
        for step_idx, (time, time_next) in enumerate(time_pairs):
            time_next = max(time_next, 0)
            if time == time_next == 0:
                continue

            ts = torch.tensor([time], device=x_t.device, dtype=torch.long)
            x0_pred, pred_noise = self.model_prediction(x_t, cond, ts)

            # 将最新输出压入 model_outputs 队列（与附件一致）:contentReference[oaicite:11]{index=11}
            for i in range(self.solver_order - 1):
                self.model_outputs[i] = self.model_outputs[i + 1]
            self.model_outputs[-1] = x0_pred  # 注意：附件里放的是 "当前步的模型输出变量"，本包装中我们将其对齐为 x0_pred

            lower_order_final = ((step_idx == len(time_pairs) - 1) and self.lower_order_final and S <= 15)
            lower_order_second = ((step_idx == len(time_pairs) - 2) and self.lower_order_final and S <= 15)

            if (self.solver_order == 1) or (self.lower_order_nums < 1) or lower_order_final:
                if self.sampler == 'dpmsolver':
                    x_t = self.dpm_solver_first_order_update(x0_pred, time, time_next, x_t)
                elif self.sampler == 'deis':
                    x_t = self.deis_first_order_update(x0_pred, time, time_next, x_t)
                else:
                    raise NotImplementedError
            elif (self.solver_order == 2) or (self.lower_order_nums < 2) or lower_order_second:
                tlist = [self.timesteps[step_idx - 1], time]
                if self.sampler == 'dpmsolver':
                    x_t = self.multistep_dpm_solver_second_order_update(self.model_outputs, tlist, time_next, x_t)
                elif self.sampler == 'deis':
                    x_t = self.multistep_deis_second_order_update(self.model_outputs, tlist, time_next, x_t)
                else:
                    raise NotImplementedError
            else:
                tlist = [self.timesteps[step_idx - 2], self.timesteps[step_idx - 1], time]
                if self.sampler == 'dpmsolver':
                    x_t = self.multistep_dpm_solver_third_order_update(self.model_outputs, tlist, time_next, x_t)
                elif self.sampler == 'deis':
                    x_t = self.multistep_deis_third_order_update(self.model_outputs, tlist, time_next, x_t)
                else:
                    raise NotImplementedError

            if self.return_intermediate:
                inter.append(x_t)
            if self.lower_order_nums < self.solver_order:
                self.lower_order_nums += 1

        return x_t, pred_noise, inter

    # ----------------- 以下 **一步/多步** 更新与系数，均与附件保持一致 ----------------- :contentReference[oaicite:12]{index=12}
    def deis_first_order_update(self, model_output, timestep, prev_timestep, sample):
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.sqrt_alphas_cumprod[prev_timestep], self.sqrt_alphas_cumprod[timestep]
        sigma_t = self.sigma_t[prev_timestep]
        h = lambda_t - lambda_s
        # log-ρ DEIS（与你附件中“only support log-rho multistep deis now”一致）
        x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        return x_t

    def dpm_solver_first_order_update(self, model_output, timestep, prev_timestep, sample):
        lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
        alpha_t, alpha_s = self.sqrt_alphas_cumprod[prev_timestep], self.sqrt_alphas_cumprod[timestep]
        sigma_t, sigma_s = self.sigma_t[prev_timestep], self.sigma_t[timestep]
        h = lambda_t - lambda_s
        # 标准 DPM-Solver（++ 变体在附件中也有；此处与 ‘dpmsolver’ 分支对齐）
        x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        return x_t

    def multistep_deis_second_order_update(self, model_output_list: List[torch.FloatTensor], timestep_list: List[int],
                                           prev_timestep: int, sample: torch.FloatTensor):
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        alpha_t, alpha_s0, alpha_s1 = self.sqrt_alphas_cumprod[t], self.sqrt_alphas_cumprod[s0], self.sqrt_alphas_cumprod[s1]
        sigma_t, sigma_s0, sigma_s1 = self.sigma_t[t], self.sigma_t[s0], self.sigma_t[s1]
        rho_t, rho_s0, rho_s1 = sigma_t / alpha_t, sigma_s0 / alpha_s0, sigma_s1 / alpha_s1

        def ind_fn(tt, b, c):
            return tt * (-torch.log(c) + torch.log(tt) - 1) / (torch.log(b) - torch.log(c))

        coef1 = ind_fn(rho_t, rho_s0, rho_s1) - ind_fn(rho_s0, rho_s0, rho_s1)
        coef2 = ind_fn(rho_t, rho_s1, rho_s0) - ind_fn(rho_s0, rho_s1, rho_s0)
        x_t = alpha_t * (sample / alpha_s0 + coef1 * m0 + coef2 * m1)
        return x_t

    def multistep_dpm_solver_second_order_update(self, model_output_list: List[torch.FloatTensor], timestep_list: List[int],
                                                 prev_timestep: int, sample: torch.FloatTensor):
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]
        alpha_t, alpha_s0 = self.sqrt_alphas_cumprod[t], self.sqrt_alphas_cumprod[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        # midpoint 版本（与附件默认一致）
        x_t = ((alpha_t / alpha_s0) * sample
               - (sigma_t * (torch.exp(h) - 1.0)) * D0
               - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1)
        return x_t

    def multistep_deis_third_order_update(self, model_output_list: List[torch.FloatTensor], timestep_list: List[int],
                                          prev_timestep: int, sample: torch.FloatTensor):
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        alpha_t, alpha_s0, alpha_s1, alpha_s2 = self.sqrt_alphas_cumprod[t], self.sqrt_alphas_cumprod[s0], self.sqrt_alphas_cumprod[s1], self.sqrt_alphas_cumprod[s2]
        sigma_t, sigma_s0, sigma_s1, sigma_s2 = self.sigma_t[t], self.sigma_t[s0], self.sigma_t[s1], self.sigma_t[s2]
        rho_t, rho_s0, rho_s1, rho_s2 = sigma_t/alpha_t, sigma_s0/alpha_s0, sigma_s1/alpha_s1, sigma_s2/alpha_s2

        def ind_fn(tt, b, c, d):
            num = tt * (torch.log(c) * (torch.log(d) - torch.log(tt) + 1) - torch.log(d) * torch.log(tt) + torch.log(d) + torch.log(tt) ** 2 - 2 * torch.log(tt) + 2)
            den = (torch.log(b) - torch.log(c)) * (torch.log(b) - torch.log(d))
            return num / den

        coef1 = ind_fn(rho_t, rho_s0, rho_s1, rho_s2) - ind_fn(rho_s0, rho_s0, rho_s1, rho_s2)
        coef2 = ind_fn(rho_t, rho_s1, rho_s2, rho_s0) - ind_fn(rho_s0, rho_s1, rho_s2, rho_s0)
        coef3 = ind_fn(rho_t, rho_s2, rho_s0, rho_s1) - ind_fn(rho_s0, rho_s2, rho_s0, rho_s1)
        x_t = alpha_t * (sample / alpha_s0 + coef1 * m0 + coef2 * m1 + coef3 * m2)
        return x_t

    def multistep_dpm_solver_third_order_update(self, model_output_list: List[torch.FloatTensor], timestep_list: List[int],
                                                prev_timestep: int, sample: torch.FloatTensor):
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        lambda_t, lambda_s0, lambda_s1, lambda_s2 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1], self.lambda_t[s2]
        alpha_t, alpha_s0 = self.sqrt_alphas_cumprod[t], self.sqrt_alphas_cumprod[s0]
        sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        x_t = ((alpha_t / alpha_s0) * sample
               - (sigma_t * (torch.exp(h) - 1.0)) * D0
               - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
               - (sigma_t * ((torch.exp(h) - 1.0 - h) / (h ** 2) - 0.5)) * D2)
        return x_t
