import torch
import math
from typing import Iterable

def run_cross_entropy(
    inputs: torch.Tensor,   # shape: [batch_size, vocab_size]，logits（未归一化）
    targets: torch.Tensor,  # shape: [batch_size]，正确类别的整数索引
) -> torch.Tensor:
    
    # 交叉熵 = -log(softmax(logits)_target) = -log(exp(logits_target) / sum(exp(logits)))
    #  = -log(exp(logits_target)) + log(sum(exp(logits))) = -logit_target + log_sum_exp


    max_logits = torch.max(inputs, dim = -1, keepdim=True).values
    inputs = inputs - max_logits
    log_sum_exp = inputs.exp().sum(dim= -1).log()
    logits_target = inputs[torch.arange(inputs.shape[0]), targets]
    loss = -logits_target + log_sum_exp
    return loss.mean()

    # 但直接算 softmax 再 log 会数值不稳定，用 log_softmax 技巧：
    #
    # 1. 沿最后一维取最大值（数值稳定化，和 softmax 同理）
    #    max_logits shape: [batch_size, 1] (因为max keepdim=True)
    # 2. inputs 减去最大值
    # 3. 算 log_sum_exp：对减完的结果先 exp，再 sum，再 log
    #    log_sum_exp shape: [batch_size]
    # 4. 对每个样本，取出 targets 指定位置的 logit（减完最大值后的版本）
    #    提示：用 inputs[range(batch_size), targets] 做高级索引
    #    gathered shape: [batch_size]
    # 5. loss = -gathered + log_sum_exp（这就是 log_softmax[target] 的简化）
    # 6. 对所有样本取平均（.mean()），return



def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    # 把所有参数的梯度整体裁剪到不超过 max_l2_norm
    
    parameters = list(parameters)

     # 1. 收集所有 requires_grad=True 的参数的 grad
     #    参数可能有 .grad = None 的情况（被冻结的参数），跳过
     # 2. 把所有 grad flatten 后拼成一个大向量 all_grads
     #    提示：用 torch.cat([g.flatten() for g in grads])
    all_grad = torch.cat([p.grad.flatten() 
                          for p in parameters if p.grad is not None])
    
     # 3. 算当前的总 L2 范数：torch.norm(all_grads, 2)
    L2 = torch.norm(all_grad, 2)
     # 4. 如果 总范数 > max_l2_norm：
     #    clip_ratio = max_l2_norm / 总范数
     #    对每个有 grad 的参数：grad *= clip_ratio（原地修改，in-place）\
    if L2 > max_l2_norm:
        clip_ratio = max_l2_norm / L2
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_ratio) # = p.grad * clip_ratio 区别在于原地修改
    # 5. 如果 总范数 <= max_l2_norm，什么都不做
    # 6. 无 return（原地修改梯度）


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:

    if it < warmup_iters:
        lr = max_learning_rate * it / warmup_iters

    elif warmup_iters <= it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(progress * math.pi))

    else:
        lr = min_learning_rate
    return lr
    # 三段式学习率：warmup → cosine decay → min_lr
    # 参考 test_optimizer.py:58-84 里的 expected_lrs 数组来理解
    #
    # 1. 如果 it < warmup_iters（warmup 阶段）：
    #    线性增长：lr = max_lr * it / warmup_iters
    #    注意 it=0 时 lr 不应该是 0，而是 max_lr / warmup_iters
    #
    # 2. 如果 warmup_iters <= it < cosine_cycle_iters（cosine 阶段）：
    #    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)  （0到1的进度）
    #    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(progress * π))
    #    提示：用 math.cos，角度转弧度
    #
    # 3. 否则（cosine 结束后）：
    #    lr = min_lr
    #
    # 4. return lr


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    # 返回一个自定义的 AdamW 优化器类，继承 torch.optim.Optimizer
    #
    class MyAdamW(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
            defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
            super().__init__(params, defaults)

        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad.data
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    # Update biased first moment estimate
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    # Update biased second raw moment estimate
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Compute bias-corrected first and second moment estimates
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                    step_size = group['lr'] / bias_correction1

                    # Update parameters
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                    # Apply weight decay directly to the parameters
                    if group['weight_decay'] != 0:
                        p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

    return MyAdamW
    # 类的结构（参考 torch.optim.Optimizer 的文档）：
    #
    # __init__(self, params, lr, weight_decay, betas, eps):
    #   1. 调用 super().__init__(params, defaults=dict(lr=lr, ...))
    #   2. 存储各超参数
    #
    # step(self, closure=None):
    #   对每个参数组（self.param_groups）中的每个参数 p：
    #   a. 如果 p.grad is None，跳过
    #   b. 从 self.state[p] 取出/初始化：步数计数 step_count、一阶矩 m、二阶矩 v
    #      step_count 初值 0，m 和 v 初值为和 p 同 shape 的零向量
    #   c. 更新 step_count += 1
    #   d. bias_correction：
    #      bias_correction1 = 1 - beta1^step_count
    #      bias_correction2 = 1 - beta2^step_count
    #   e. 更新一阶矩：m = beta1 * m + (1 - beta1) * grad
    #   f. 更新二阶矩：v = beta2 * v + (1 - beta2) * grad^2（element-wise 平方）
    #   g. m_hat = m / bias_correction1
    #      v_hat = v / bias_correction2
    #   h. p.data -= lr * m_hat / (sqrt(v_hat) + eps)    ← Adam 的更新
    #   i. weight decay：p.data -= lr * weight_decay * p.data  ← 原始 W 的 decay
    #      注意：weight decay 是直接作用在 p.data 上的，不是作用在 grad 上
    #
    # return 你的类
    


def run_save_checkpoint(model, optimizer, iteration, out):
    # 用 torch.save 把三样东西存到一个字典里
    # model.state_dict(), optimizer.state_dict(), iteration
    # 存到 out（可能是路径也可能是文件对象）
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)


def run_load_checkpoint(src, model, optimizer):
    # 用 torch.load 从 src 加载
    # model.load_state_dict(...)
    # optimizer.load_state_dict(...)
    # return iteration
    torch_load = torch.load(src)
    model.load_state_dict(torch_load['model_state_dict'])
    optimizer.load_state_dict(torch_load['optimizer_state_dict'])
    return torch_load['iteration']