import os
import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import argparse
import diffuser.datasets as datasets
import diffuser.utils as utils

from loguru import logger

from tqdm import tqdm

os.chdir('/root/diffuser')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Maze 2D Table')
    parser.add_argument('--config', type=str, default='config.maze2d', help='Path to the configuration file')
    parser.add_argument('--dataset', type=str, default='maze2d-umaze-v1', help='Dataset to use')
    parser.add_argument('--numepisodes', type=int, default=5, help='Number of episodes to run')
    return parser.parse_args()



if __name__=="__main__":
    args = parse_arguments()
    config_path = args.config
    dataset = args.dataset
    N = args.numepisodes
    pretrainedpath = 'logs_2'
    savepath = join(pretrainedpath, f'{dataset}/rollouts')
    logger.info(f"Creating dir {savepath}: {utils.mkdir(savepath)})")
    vis_freq = 10
    batch_size = 1
    dataset = 'maze2d-large-v1'

    env = datasets.load_environment(dataset)

    if dataset=='maze2d-umaze-v1':
        diffusion_experiment = utils.load_diffusion(pretrainedpath, dataset, 'diffusion/H128_T64', epoch='latest')
    elif dataset=='maze2d-medium-v1':
        diffusion_experiment = utils.load_diffusion(pretrainedpath, dataset, 'diffusion/H256_T256', epoch='latest')
    elif dataset=='maze2d-large-v1':
        diffusion_experiment = utils.load_diffusion(pretrainedpath, dataset, 'diffusion/H384_T256', epoch='latest')
    else:
        raise ValueError(f"Dataset {dataset} not found")

    diffusion = diffusion_experiment.ema
    dataset = diffusion_experiment.dataset
    renderer = diffusion_experiment.renderer

    policy = Policy(diffusion, dataset.normalizer)

    isconditioned = True

    scorelist = []
    for i in tqdm(range(N)):
        savepath_iter = join(savepath, f'iter_{i}')
        logger.info(f"Creating dir {savepath_iter}: {utils.mkdir(savepath_iter)})")
        observation = env.reset()

        if isconditioned:
            logger.info('Resetting target')
            env.set_target()

        ## set conditioning xy position to be the goal
        target = env._target
        cond = {
            diffusion.horizon - 1: np.array([*target, 0, 0]),
        }

        ## observations for rendering
        rollout = [observation.copy()]

        total_reward = 0
        for t in range(env.max_episode_steps):

            state = env.state_vector().copy()

            ## can replan if desired, but the open-loop plans are good enough for maze2d
            ## that we really only need to plan once
            if t == 0:
                cond[0] = observation

                action, samples = policy(cond, batch_size=batch_size)
                actions = samples.actions[0]
                sequence = samples.observations[0]
            # pdb.set_trace()

            # ####
            if t < len(sequence) - 1:
                next_waypoint = sequence[t+1]
            else:
                next_waypoint = sequence[-1].copy()
                next_waypoint[2:] = 0

            ## can use actions or define a simple controller based on state predictions
            action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

            next_observation, reward, terminal, _ = env.step(action)
            total_reward += reward
            score = env.get_normalized_score(total_reward)
            # logger.info(
            #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | ref_max_score: {env.ref_max_score} | ref_min_score: {env.ref_min_score}'
            #     f'{action}'
            # )

            xy = next_observation[:2]
            goal = env.unwrapped._target
            # logger.info(f'maze | pos: {xy} | goal: {goal}')

            ## update rollout observations
            rollout.append(next_observation.copy())

            if t % vis_freq == 0 or terminal:
                fullpath = join(savepath_iter, f'{t}.png')

                if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)



                ## save rollout thus far
                renderer.composite(join(savepath_iter, 'rollout.png'), np.array(rollout)[None], ncol=1)



            if terminal:
                break

            observation = next_observation
        scorelist.append((total_reward, score)) 

    # Save scorelist as json
    result = {
        'return': [item[0] for item in scorelist],
        'scores': [item[1]*100 for item in scorelist]
    }
    json_path = join(savepath, 'scorelist.json')
    with open(json_path, 'w') as f:
        json.dump(result, f)
    logger.info(f"Scorelist saved to {json_path}")
