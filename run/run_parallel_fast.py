from os.path import exists
from pathlib import Path
import uuid
#from josh_env import RedGymEnv
#from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from src.tensorboard_callback import TensorboardCallback

from src.globals import env_config, ep_length, batch_size, n_epochs
from src.red_environment import RedGymEnv

def make_env(env_conf={}, reward_hyperparams={}, seed=0):
    def _init():
        env = RedGymEnv(config=env_conf,reward_hyperparameters=reward_hyperparams)
        env.reset(seed=seed)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    use_wandb_logging = False
    
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'storage/sessions/session_{sess_id}')
    env_config['session_path'] = sess_path
    
    num_cpu = 16  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='poke')
    
    callbacks = [checkpoint_callback, TensorboardCallback()]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())

    #env_checker.check_env(env)
    learn_steps = 40
    # put a checkpoint here you want to start from
    file_name = 'session_e41c9eff/poke_38207488_steps' 
    
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        print('\nCreating new model')
        model = PPO('CnnPolicy', env, verbose=2, n_steps=ep_length // 8, batch_size=batch_size, n_epochs=n_epochs, gamma=0.998, tensorboard_log=sess_path)
    
    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu*1000, callback=CallbackList(callbacks))

    #env_infos = []
    #for i in range(num_cpu):
        #env_infos.append(env.get_attr('get_info')[i])

    if use_wandb_logging:
        run.finish()