import gym
import d4rl
import numpy as np
import os

# 1. 设置路径
save_path = "/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_clk/ReinFlow/data/gym/kitchen-complete-v0/normalization.npz"
env_name = "kitchen-complete-v0"

print(f"正在修复 {env_name} ...")

try:
    # 2. 加载数据
    env = gym.make(env_name)
    dataset = env.get_dataset()
    obs = dataset['observations']
    actions = dataset['actions']

    # 3. 计算所有统计量 (必须包含 mean 和 std)
    mean = np.mean(obs, axis=0).astype(np.float32)
    std = np.std(obs, axis=0).astype(np.float32) + 1e-3
    
    obs_min = np.min(obs, axis=0).astype(np.float32)
    obs_max = np.max(obs, axis=0).astype(np.float32)
    
    action_mean = np.mean(actions, axis=0).astype(np.float32)
    action_std = np.std(actions, axis=0).astype(np.float32) + 1e-3
    action_min = np.min(actions, axis=0).astype(np.float32)
    action_max = np.max(actions, axis=0).astype(np.float32)

    # 4. 保存 (确保包含所有键)
    np.savez(save_path, 
             mean=mean,          # <--- 之前缺了这个
             std=std,            # <--- 之前缺了这个
             obs_min=obs_min, 
             obs_max=obs_max,
             action_mean=action_mean,
             action_std=action_std,
             action_min=action_min,
             action_max=action_max
             )
             
    print(f"✅ 修复成功！文件已保存至: {save_path}")
    
    # 5. 立即验证
    data = np.load(save_path)
    print("当前文件包含的键:", list(data.keys()))
    if 'mean' in data and data['mean'].shape == (17,):
        print("验证通过：mean 存在且维度正确 (17,)")
    else:
        print("验证失败！")

except Exception as e:
    print(f"❌ 错误: {e}")