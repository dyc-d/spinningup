import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
'''
# 这个函数的主要作用是用来构建一个前馈神经网络，其中输入参数 sizes 表示每层神经网络的大小，例如 [4, 32, 2] 表示输入层有4个神经元，隐藏层有32个神经元，输出层有2个神经元。
# activation 和 output_activation 分别表示隐层和输出层的激活函数，如果没有指定则默认为 nn.Tanh 和 nn.Identity。
# 在函数体中，我们首先定义了一个空的列表 layers 用于存储神经网络的每一层。然后通过一个循环来遍历每一层神经网络，根据神经网络层数的不同选择相应的激活函数 act。在每一层中，
# 我们添加了一个线性层 nn.Linear(sizes[j], sizes[j+1])，其作用是对输入进行线性变换并输出到下一层，然后接上激活函数 act()，将输出值进行非线性变换。
# 最后，我们将这些层按照顺序组合成一个nn.Sequential 模型，该模型按照添加的顺序执行每一层，并将输出传递到下一层，最终得到模型的输出。该模型可以用于各种机器学习任务，
# 如分类、回归等。
'''
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
'''
# hidden_sizes：神经网络隐藏层的大小，是一个整数列表。
# lr：学习率。
# epochs：训练的轮数。
# batch_size：每个batch的样本数量。
# render：是否在训练时渲染环境。
'''
def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    # 这行代码是在检查环境的状态空间类型是否为连续的，如果不是则会抛出异常并且输出一段错误提示信息。其中“Box”是一个状态空间类型，它表示的是连续的实数空间。
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."
    # 在这个CartPole环境中，env.observation_space状态是一个长度为4的一维数组，代表了小车位置，小车速度，杆子角度和杆子角速度等4个特征。
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    # 这个函数定义了如何得到一个由神经网络参数化的策略分布（policy distribution）。具体来说，输入观测 obs，经过神经网络 logits_net 的变换得到一个对应的输出 logits，
    # 然后利用 PyTorch中提供的Categorical类将logits转化为一个以softmax(logits)为概率分布的Categorical 对象，最后返回这个对象,例如返回一个动作的概率分布probs = [0.3, 0.7]。
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    # 首先调用 get_policy(obs) 获取当前状态下的动作分布（即策略），然后使用 log_prob(act) 方法计算当前动作采样自该分布的对数概率，
    # 并乘以 weights 得到加权对数概率。最后，将加权对数概率取平均，再取负号，得到损失函数
    ## 其中求平均是因为使用了批量更新即一个batch，然后权重的基于重要性采用得到的
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                # 这里的 * 不是乘法而是复制ep_len个
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
