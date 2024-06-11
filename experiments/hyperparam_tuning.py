from ast import Sub
from os.path import exists
from pathlib import Path
import uuid
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from src.tensorboard_callback import TensorboardCallback
from alive_progress import alive_bar
from run.run_parallel_fast import make_env
import os
import datetime as dt

class BaseRun:
    def __init__(self, env_config, ep_length, batch_size, n_epochs, hyperparameter_config, value_function, checkpoint_file = None):
        self.env_config = env_config
        self.hyperparameter_config = hyperparameter_config
        self.value_function = value_function
        self.ep_length = ep_length
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learn_steps = 2
        self.num_cpu = 6
        self.checkpoint_file = checkpoint_file

    def execute_run(self) -> float:
        sess_id = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")#str(uuid.uuid4())[:8]
        if not os.path.exists('storage/sessions/base'):
            os.mkdir('storage/sessions/base')
        sess_path = Path(f'storage/sessions/base/session_{sess_id}')
        self.env_config['session_path'] = sess_path
        
        env = SubprocVecEnv([make_env(env_conf=self.env_config,reward_hyperparams=self.hyperparameter_config,seed=i) for i in range(self.num_cpu)])
        
        checkpoint_callback = CheckpointCallback(save_freq=self.ep_length, save_path=sess_path,
                                        name_prefix='poke')
        
        callbacks = [checkpoint_callback, TensorboardCallback()]
        if exists(str(self.checkpoint_file) + '.zip'):
            print('\nloading checkpoint')
            model = PPO.load(self.checkpoint_file, env=env)
            model.n_steps = self.ep_length
            model.n_envs = self.num_cpu
            model.rollout_buffer.buffer_size = self.ep_length
            model.rollout_buffer.n_envs = self.num_cpu
            model.rollout_buffer.reset()
        else:
            print("Skipping checkpoint")
            model = PPO('CnnPolicy', env, verbose=2, n_steps=self.ep_length // 8, batch_size=self.batch_size, n_epochs=self.n_epochs, gamma=0.9999, tensorboard_log=sess_path)
        
        for i in range(self.learn_steps):
            scaleup = 1
            n_steps = (self.ep_length)*self.num_cpu*scaleup
            print("Running for this many steps: ", (self.ep_length)*self.num_cpu*scaleup)
            model.learn(total_timesteps=n_steps, callback=CallbackList(callbacks))
        return self.env_config['session_path']



class Tuner:
    def __init__(self, env_config, ep_length, batch_size, n_epochs, hyperparameter_config, value_function, checkpoint_file: str = None):
        self.env_config = env_config
        self.hyperparameter_config = hyperparameter_config
        print(f'hyperparameter_config: {hyperparameter_config}')
        self.value_function = value_function
        self.ep_length = ep_length
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learn_steps = 2
        self.num_cpu = 1
        self.checkpoint_file = checkpoint_file

    
    def execute_run(self) -> float:
        sess_id = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
        if not os.path.exists('storage/sessions/tunes'):
            os.mkdir('storage/sessions/tunes')
        sess_path = Path(f'storage/sessions/tunes/session_{sess_id}')
        self.env_config['session_path'] = sess_path
        
        #env = SubprocVecEnv([make_env(env_conf=self.env_config,reward_hyperparams=self.hyperparameter_config,seed=i) for i in range(self.num_cpu)])
        env = make_env(self.env_config, self.hyperparameter_config)()
        checkpoint_callback = CheckpointCallback(save_freq=self.ep_length, save_path=sess_path,
                                        name_prefix='poke')
        
        callbacks = [checkpoint_callback, TensorboardCallback()]
        # put a checkpoint here you want to start from
        #file_name = 'session_e41c9eff/poke_38207488_steps' 
        print(f"Destination: {self.checkpoint_file}")
        if exists(str(self.checkpoint_file) + '.zip'):
            print('\nloading checkpoint')
            model = PPO.load(self.checkpoint_file, env=env)
            model.n_steps = self.ep_length
            model.n_envs = self.num_cpu
            model.rollout_buffer.buffer_size = self.ep_length
            model.rollout_buffer.n_envs = self.num_cpu
            model.rollout_buffer.reset()
        else:
            print('\nCreating new model')
            model = PPO('CnnPolicy', env, verbose=2, n_steps=self.ep_length // 8, batch_size=self.batch_size, n_epochs=self.n_epochs, gamma=0.9999, tensorboard_log=sess_path)
        
        for i in range(self.learn_steps):
            scaleup = 1
            n_steps = (self.ep_length)*self.num_cpu*scaleup
            print("Running for this many steps: ", (self.ep_length)*self.num_cpu*scaleup)
            model.learn(total_timesteps=n_steps, callback=CallbackList(callbacks))
        return self.value_function([env.get_info()])
        env_infos = []
        print(dir(env))
        print(env.metadata)
        print(env.processes)
        print(env.num_envs)
        raise Exception("Testing")
        for i in range(self.num_cpu):
            print(env[i].get_attr('get_info'))
            env_infos.append(env.get_attr('get_info'))
        return self.value_function(env_infos)
        
