import numpy as np
from rlgym_sim.envs import Match
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

# from stable_baselines3.ppo import MlpPolicy

import torch as T

from wandb.integration.sb3 import WandbCallback
import wandb

from rlgym_sim.utils.obs_builders import AdvancedObs
from rlgym_sim.utils.state_setters import RandomState

from rlgym_sim.utils.terminal_conditions.common_conditions import (
    # NoTouchTimeoutCondition,
    TimeoutCondition,
)
from extra_utils.goal_scored_reward import GoalScoredReward
from extra_utils.terminal_condition import GoalScoredCondition
from sb3_multi_inst_env import SB3MultipleInstanceEnv
from helpers.lookup_action import LookupAction
from helpers.extra_state_setters import (
    GoaliePracticeState,
    WeightedSampleSetter,
    WallPracticeState,
)


def main():
    frame_skip = 6
    half_life_seconds = 8

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    agents_per_match = 4
    num_instances = 5
    batch_size = 100000
    target_steps = 2_500_000
    steps = target_steps // (num_instances * agents_per_match)

    # if steps % batch_size != 0:
    #     print(
    #         "Please assign proper value for 'target_steps'",
    #         "so steps is divisible by batch_size to maximize efficiency",
    #     )

    training_interval = 25_000_000
    mmr_save_frequency = 50_000_000

    model_save_dir = "./models"
    model_save_path = model_save_dir + "/exit_save.zip"

    def exit_save(model):
        model.save(model_save_path)

    def get_match():
        return Match(
            team_size=2,
            reward_function=GoalScoredReward(),
            spawn_opponents=True,
            terminal_conditions=[
                # NoTouchTimeoutCondition(800),
                TimeoutCondition(500),
                GoalScoredCondition(),
            ],
            obs_builder=AdvancedObs(),
            state_setter=WeightedSampleSetter(
                (
                    RandomState(True, True, False),
                    GoaliePracticeState(
                        allow_enemy_interference=True,
                    ),
                    WallPracticeState(0.5, 0.2, 0.3),
                ),
                (0.2, 0.5, 0.3),
            ),
            # state_setter=RandomState(True, True, False),
            # state_setter=GoaliePracticeState(allow_enemy_interference=True),
            action_parser=LookupAction(),  # Lookup > Discrete
        )

    env = SB3MultipleInstanceEnv(
        get_match,
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
            },  # automatically adjusts to users changing instance count, may encounter shaping error otherwise
            # If you need to adjust parameters mid training, you can use the below example as a guide
            # custom_objects={"n_envs": env.num_envs, "n_steps": steps, "batch_size": batch_size, "n_epochs": 10, "learning_rate": 5e-5}
        )
        print("Loaded previous exit save.")
    except Exception as e:
        print("No saved model found, error : ", e)
        print("Exitting...")
        env.close()
        # return
        # from torch.nn import Tanh as act_fn

        # policy_kwargs = dict(
        #     activation_fn=act_fn,
        #     net_arch=dict(pi=[512, 256, 256], vf=[512, 512, 512, 512]),
        # )

        # model = PPO(
        #     MlpPolicy,
        #     env,
        #     n_epochs=10,  # PPO calls for multiple epochs
        #     policy_kwargs=policy_kwargs,
        #     learning_rate=5e-5,  # Around this is fairly common for PPO
        #     ent_coef=0.01,  # From PPO Atari
        #     vf_coef=0.8,  # From PPO Atari
        #     gamma=gamma,  # Gamma as calculated using half-life
        #     verbose=3,  # Print out all the info as we're going
        #     batch_size=batch_size,  # Batch size as high as possible within reason
        #     n_steps=steps,  # Number of steps to perform before optimizing network
        #     tensorboard_log="logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
        #     device="cpu",  # Uses GPU if available
        # )

    wandb.init(
        project="mercury-bot",
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
                callback=ckpt_callback,
                reset_num_timesteps=False,
                tb_log_name="Mercury",
            )
            model.save(model_save_path)
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency

    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    exit_save(model)
    print("Save complete")
    wandb.finish()


if __name__ == "__main__":  # Required for multiprocessing
    main()
