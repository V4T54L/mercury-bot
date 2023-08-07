import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

# from stable_baselines3.ppo import MlpPolicy

import torch as T

from wandb.integration.sb3 import WandbCallback
import wandb

from sb3_multi_inst_env import SB3MultipleInstanceEnv

from helpers.matches import (
    get_random_goalie_match,
    get_replay_setter_match,
    get_random_match,
)


def main():
    frame_skip = 6
    half_life_seconds = 8

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    agents_per_match = 4
    num_instances = 3
    batch_size = 100000
    target_steps = 1_000_000
    steps = target_steps // (num_instances * agents_per_match)

    training_interval = 25_000_000
    mmr_save_frequency = 50_000_000

    model_save_dir = "./models"
    model_save_path = model_save_dir + "/exit_save.zip"
    replay_file_path = "../NPZ_Data/states_scores_doubles.npz"

    def exit_save(model):
        model.save(model_save_path)

    def get_matches_list(num_instances, percentage_of_replay_setters):
        if not 0 <= percentage_of_replay_setters <= 1:
            raise ValueError(
                "Value mustbe between 0 and 1 for 'percentage_of_replay_setters'"
            )
        len_replay_states = 787254
        n_replay_matches = int(num_instances * percentage_of_replay_setters)
        replay_batch_size = len_replay_states // n_replay_matches
        starts = [i for i in range(0, len_replay_states, replay_batch_size)]

        return [
            get_replay_setter_match(
                replay_file_path, starts[i], starts[i] + replay_batch_size
            )
            if i < n_replay_matches
            # else get_random_goalie_match()
            else get_random_match()
            for i in range(num_instances)
        ]

    env = SB3MultipleInstanceEnv(
        get_matches_list(num_instances, 0.3),
        num_instances,
        wait_time=0,
        tick_skip=8,
        dodge_deadzone=0.5,
        copy_gamestate_every_step=False,
    )
    env = VecCheckNan(env)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, gamma=gamma)

    try:
        device = T.device("cuda" if T.cuda.is_available() else "cpu")
        model = PPO.load(
            model_save_path,
            env,
            device=device,
            custom_objects={
                "n_envs": env.num_envs,
                "n_steps": steps,
                "batch_size": batch_size,
                "learning_rate": 5e-5,
                "vf_coef": 0.7,
            },
        )
        print("Loaded previous exit save.")
    except Exception as e:
        print("No saved model found, error : ", e)
        print("Exitting...")
        env.close()

    wandb.tensorboard.patch(root_logdir="logs/Mercury_0")
    wandb.init(
        project="mercury-bot",
        job_type="training-mercury-bot",
        sync_tensorboard=True,
    )

    ckpt_callback = CheckpointCallback(
        round(5_000_000 / env.num_envs),
        save_path=model_save_dir,
        name_prefix="rl_model",
    )

    callback_list = CallbackList([ckpt_callback, WandbCallback()])

    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            model.learn(
                training_interval,
                callback=callback_list,
                reset_num_timesteps=False,
                tb_log_name="Mercury",
            )
            model.save(model_save_path)
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency

    except KeyboardInterrupt:
        print("Exiting training")

    except Exception as e:
        print("Uncaught exception :", e)

    print("Saving model")
    exit_save(model)
    print("Save complete")
    wandb.finish()


if __name__ == "__main__":  # Required for multiprocessing
    main()
