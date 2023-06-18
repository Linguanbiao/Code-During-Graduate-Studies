 
import os
import math

import numpy as np
import torch
import opacus.privacy_analysis as tf_privacy

ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def get_renyi_divergence(sample_rate, noise_multiplier, orders=ORDERS):
    rdp = torch.tensor(
        tf_privacy.compute_rdp(
            sample_rate, noise_multiplier, 1, orders
        )
    )
    return rdp


def get_privacy_spent(total_rdp, target_delta=1e-5, orders=ORDERS):
    return tf_privacy.get_privacy_spent(orders, total_rdp, target_delta)


def get_epsilon(sample_rate, mul, num_steps, target_delta=1e-5, orders=ORDERS, rdp_init=0):
    # compute the epsilon budget spent after `num_steps` with batch sampling rate
    # of `sample_rate` and a noise multiplier of `mul`

    rdp = rdp_init + get_renyi_divergence(sample_rate, mul, orders=orders) * num_steps
    eps, _ = get_privacy_spent(rdp, target_delta=target_delta, orders=orders)
    return eps


def get_noise_mul(num_samples, batch_size, target_epsilon, epochs, rdp_init=0, target_delta=1e-5, orders=ORDERS):
    # compute the noise multiplier that results in a privacy budget
    # of `target_epsilon` being spent after a given number of epochs of DP-SGD.

    mul_low = 100
    mul_high = 0.1

    num_steps = math.floor(num_samples // batch_size) * epochs
    sample_rate = batch_size / (1.0 * num_samples)

    eps_low = get_epsilon(sample_rate, mul_low, num_steps, target_delta, orders, rdp_init=rdp_init)
    eps_high = get_epsilon(sample_rate, mul_high, num_steps, target_delta, orders, rdp_init=rdp_init)

    assert eps_low < target_epsilon
    assert eps_high > target_epsilon

    while eps_high - eps_low > 0.01:
        mul_mid = (mul_high + mul_low) / 2
        eps_mid = get_epsilon(sample_rate, mul_mid, num_steps, target_delta, orders, rdp_init=rdp_init)

        if eps_mid <= target_epsilon:
            mul_low = mul_mid
            eps_low = eps_mid
        else:
            mul_high = mul_mid
            eps_high = eps_mid

    return mul_low


def get_noise_mul_privbyiter(num_samples, batch_size, target_epsilon, epochs, target_delta=1e-5):
    mul_low = 100
    mul_high = 0.1

    eps_low = priv_by_iter_guarantees(epochs, batch_size, num_samples, mul_low, target_delta, verbose=False)
    eps_high = priv_by_iter_guarantees(epochs, batch_size, num_samples, mul_high, target_delta, verbose=False)

    assert eps_low < target_epsilon
    assert eps_high > target_epsilon

    while eps_high - eps_low > 0.01:
        mul_mid = (mul_high + mul_low) / 2
        eps_mid = priv_by_iter_guarantees(epochs, batch_size, num_samples, mul_mid, target_delta, verbose=False)

        if eps_mid <= target_epsilon:
            mul_low = mul_mid
            eps_low = eps_mid
        else:
            mul_high = mul_mid
            eps_high = eps_mid

    return mul_low


def scatter_normalization(train_loader, scattering, K, device,
                          data_size, sample_size,
                          noise_multiplier=1.0, orders=ORDERS, save_dir=None):
    # privately compute the mean and variance of scatternet features to normalize
    # the data.

    rdp = 0
    epsilon_norm = np.inf
    if noise_multiplier > 0:
        # compute the RDP spent in this step
        sample_rate = sample_size / (1.0 * data_size)
        rdp = 2*get_renyi_divergence(sample_rate, noise_multiplier, orders)
        epsilon_norm, _ = get_privacy_spent(rdp)

    # try loading pre-computed stats
    use_scattering = scattering is not None
    assert use_scattering
    mean_path = os.path.join(save_dir, f"mean_bn_{sample_size}_{noise_multiplier}_{use_scattering}.npy")
    var_path = os.path.join(save_dir, f"var_bn_{sample_size}_{noise_multiplier}_{use_scattering}.npy")

    print(f"Using BN stats for {sample_size}/{data_size} samples")
    print(f"With noise_mul={noise_multiplier}, we get ε_norm = {epsilon_norm:.3f}")

    try:
        print(f"loading {mean_path}")
        mean = np.load(mean_path)
        var = np.load(var_path)
        print(mean.shape, var.shape)
    except OSError:

        # compute the scattering transform and the mean and squared mean of features
        scatters = []
        mean = 0
        sq_mean = 0
        count = 0
        for idx, (data, target) in enumerate(train_loader):
            with torch.no_grad():
                data = data.to(device)
                if scattering is not None:
                    data = scattering(data).view(-1, K, data.shape[2]//4, data.shape[3]//4)
                if noise_multiplier == 0:
                    data = data.reshape(len(data), K, -1).mean(-1)
                    mean += data.sum(0).cpu().numpy()
                    sq_mean += (data**2).sum(0).cpu().numpy()
                else:
                    scatters.append(data.cpu().numpy())

                count += len(data)
                if count >= sample_size:
                    break

        if noise_multiplier > 0:
            scatters = np.concatenate(scatters, axis=0)
            scatters = np.transpose(scatters, (0, 2, 3, 1))

            scatters = scatters[:sample_size]

            # s x K
            scatter_means = np.mean(scatters.reshape(len(scatters), -1, K), axis=1)
            norms = np.linalg.norm(scatter_means, axis=-1)

            # technically a small privacy leak, sue me...
            thresh_mean = np.quantile(norms, 0.5)
            scatter_means /= np.maximum(norms / thresh_mean, 1).reshape(-1, 1)
            mean = np.mean(scatter_means, axis=0)

            mean += np.random.normal(scale=thresh_mean * noise_multiplier,
                                     size=mean.shape) / sample_size

            # s x K
            scatter_sq_means = np.mean((scatters ** 2).reshape(len(scatters), -1, K),
                                       axis=1)
            norms = np.linalg.norm(scatter_sq_means, axis=-1)

            # technically a small privacy leak, sue me...
            thresh_var = np.quantile(norms, 0.5)
            print(f"thresh_mean={thresh_mean:.2f}, thresh_var={thresh_var:.2f}")
            scatter_sq_means /= np.maximum(norms / thresh_var, 1).reshape(-1, 1)
            sq_mean = np.mean(scatter_sq_means, axis=0)
            sq_mean += np.random.normal(scale=thresh_var * noise_multiplier,
                                        size=sq_mean.shape) / sample_size
            var = np.maximum(sq_mean - mean ** 2, 0)
        else:
            mean /= count
            sq_mean /= count
            var = np.maximum(sq_mean - mean ** 2, 0)

        if save_dir is not None:
            print(f"saving mean and var: {mean.shape} {var.shape}")
            np.save(mean_path, mean)
            np.save(var_path, var)

    mean = torch.from_numpy(mean).to(device)
    var = torch.from_numpy(var).to(device)

    return (mean, var), rdp


def priv_by_iter_guarantees(epochs, batch_size, samples, noise_multiplier, delta=1e-5, verbose=True):
    """Tabulating position-dependent privacy guarantees."""
    if noise_multiplier == 0:
        if verbose:
            print('No differential privacy (additive noise is 0).')
        return np.inf

    if verbose:
        print('In the conditions of Theorem 34 (https://arxiv.org/abs/1808.06651) '
              'the training procedure results in the following privacy guarantees.')
        print('Out of the total of {} samples:'.format(samples))

    steps_per_epoch = samples // batch_size
    orders = np.concatenate([np.linspace(2, 20, num=181), np.linspace(20, 100, num=81)])
    for p in (.5, .9, .99):
        steps = math.ceil(steps_per_epoch * p)  # Steps in the last epoch.
        coef = 2 * (noise_multiplier)**-2 * (
            # Accounting for privacy loss
            (epochs - 1) / steps_per_epoch +  # ... from all-but-last epochs
            1 / (steps_per_epoch - steps + 1))  # ... due to the last epoch
        # Using RDP accountant to compute eps. Doing computation analytically is
        # an option.
        rdp = [order * coef for order in orders]
        eps, _ = get_privacy_spent(rdp, delta, orders)
        if verbose:
            print('\t{:g}% enjoy at least ({:.2f}, {})-DP'.format(
                p * 100, eps, delta))

    return eps

if __name__ == '__main__':
    num_samples = 60000
    batch_size = 1024
    target_epsilon = 2.7
    target_delta = 1e-5
    epochs = 40
    noise_mul = get_noise_mul(num_samples, batch_size, target_epsilon, epochs, target_delta=target_delta)
    print(noise_mul)
    # import torchvision.datasets as dst
    # tr = dst.CIFAR100('dataset', train=True, download=True)
    # te = dst.CIFAR100('dataset', train=False, download=True)


# import turtle as t  #as就是取个别名，后续调用的t都是turtle
# from turtle import *
# import random as r
# import time
 
 
 
# n = 100.0
 
# speed("fastest")  #定义速度
# screensize(bg='black')  #定义背景颜色，可以自己换颜色
# left(90)
# forward(3*n)
# color("orange", "yellow")#定义最上端星星的颜色，外圈是orange，内部是yellow
# begin_fill()
# left(126)
 
# for i in range(5): #画五角星
#     forward(n/5)
#     right(144)    #五角星的角度
#     forward(n/5)
#     left(72)    #继续换角度
# end_fill()
# right(126)
 
# def drawlight():#定义画彩灯的方法
#     if r.randint(0, 30) == 0:#如果觉得彩灯太多，可以把取值范围加大一些，对应的灯就会少一些
#         color('tomato')#定义第一种颜色
#         circle(6)#定义彩灯大小
#     elif r.randint(0,30) == 1:
#         color('orange')#定义第二种颜色
#         circle(3)#定义彩灯大小
#     else:
#         color('dark green')#其余的随机数情况下画空的树枝
 
 
# color("dark green")#定义树枝的颜色
# backward(n*4.8)
# def tree(d, s):#开始画树
#     if d <= 0: return
#     forward(s)
#     tree(d-1, s*.8)
#     right(120)
#     tree(d-3, s*.5)
#     drawlight()#同时调用小彩灯的方法
#     right(120)
#     tree(d-3, s*.5)
#     right(120)
#     backward(s)
# tree(15, n)
# backward(n/2)
 
# for i in range(200):#循环画最底端的小装饰
#     a = 200 - 400 * r.random()
#     b = 10 - 20 * r.random()
#     up()
#     forward(b)
#     left(90)
#     forward(a)
#     down()
#     if r.randint(0, 1) == 0:
#             color('tomato')
#     else:
#         color('wheat')
#     circle(2)
#     up()
#     backward(a)
#     right(90)
#     backward(b)
 
# t.color("dark red","red")#定义字体颜色
# t.write("Merry Christmas",align ="center",font=("Comic Sans MS",40,"bold"))#定义文字、位置、字体、大小
 
 
# def drawsnow():#定义画雪花的方法
#     t.ht()  #隐藏笔头，ht=hideturtle
#     t.pensize(2)  #定义笔头大小
#     for i in range(200): #画多少雪花
#         t.pencolor("white") #定义画笔颜色为白色，其实就是雪花为白色
#         t.pu() #提笔，pu=penup
#         t.setx(r.randint(-350,350)) #定义x坐标，随机从-350到350之间选择
#         t.sety(r.randint(-100,350)) #定义y坐标，注意雪花一般在地上不会落下，所以不会从太小的纵座轴开始
#         t.pd() #落笔，pd=pendown
#         dens = 6 #雪花瓣数设为6
#         snowsize = r.randint(1,10) #定义雪花大小
#         for j in range(dens): #就是6，那就是画5次，也就是一个雪花五角星
#             #t.forward(int(snowsize))  #int（）取整数
#             t.fd(int(snowsize))
#             t.backward(int(snowsize))
#             #t.bd(int(snowsize))  #注意没有bd=backward，但有fd=forward，小bug
#             t.right(int(360/dens))  #转动角度
 
# drawsnow()#调用画雪花的方法
# t.done()  # 完成,否则会直接关闭