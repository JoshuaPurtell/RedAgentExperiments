from os.path import exists
from pathlib import Path
import uuid
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from src.tensorboard_callback import TensorboardCallback

from run.run_parallel_fast import make_env


class Tuner:
    def __init__(self, env_config, ep_length, batch_size, n_epochs, hyperparameter_config, value_function):
        self.env_config = env_config
        self.hyperparameter_config = hyperparameter_config
        self.value_function = value_function
        self.ep_length = ep_length
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    
    def execute_run(self) -> float:
        sess_id = str(uuid.uuid4())[:8]
        sess_path = Path(f'storage/sessions/session_{sess_id}')
        self.env_config['session_path'] = sess_path
        
        num_cpu = 16  # Also sets the number of episodes per training iteration
        env = SubprocVecEnv([make_env(i, self.env_config) for i in range(num_cpu)])
        
        checkpoint_callback = CheckpointCallback(save_freq=self.ep_length, save_path=sess_path,
                                        name_prefix='poke')
        
        callbacks = [checkpoint_callback, TensorboardCallback()]
        #env_checker.check_env(env)
        learn_steps = 40
        # put a checkpoint here you want to start from
        file_name = 'session_e41c9eff/poke_38207488_steps' 
        
        if exists(file_name + '.zip'):
            print('\nloading checkpoint')
            model = PPO.load(file_name, env=env)
            model.n_steps = self.ep_length
            model.n_envs = num_cpu
            model.rollout_buffer.buffer_size = self.ep_length
            model.rollout_buffer.n_envs = num_cpu
            model.rollout_buffer.reset()
        else:
            print('\nCreating new model')
            model = PPO('CnnPolicy', env, verbose=2, n_steps=self.ep_length // 8, batch_size=self.batch_size, n_epochs=self.n_epochs, gamma=0.998, tensorboard_log=sess_path)
        
        for i in range(learn_steps):
            model.learn(total_timesteps=(self.ep_length)*num_cpu*1000, callback=CallbackList(callbacks))

        env_infos = []
        for i in range(num_cpu):
            env_infos.append(env.get_attr('get_info')[i])

        return self.value_function(env_infos)
        