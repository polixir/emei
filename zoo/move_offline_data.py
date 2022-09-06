import os
import shutil
import argparse

ZOO_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ZOO_DIR)
ZOO_EXP_DIR = os.path.join(ROOT_DIR, "zoo_exp")

FILE_NEEDED = ['random.h5', 'uniform.h5', 'medium.h5', 'expert.h5', 'medium-replay.h5', 'expert-replay.h5',
               'sampling_info.txt', 'medium-agent.pth', 'expert-agent.pth']
DIR_NEEDED = ['tb']


def main(target_dir,
         source_dir=ZOO_EXP_DIR,
         algorithm="SAC",
         exp_name="default",
         file_needed=FILE_NEEDED,
         dir_needed=DIR_NEEDED):
    all_exp_dir = os.path.join(source_dir, algorithm, exp_name)
    for env in os.listdir(all_exp_dir):
        for params in os.listdir(os.path.join(all_exp_dir, env)):
            for time in os.listdir(os.path.join(all_exp_dir, env, params)):
                source_data_dir = os.path.join(all_exp_dir, env, params, time)
                contents = os.listdir(source_data_dir)
                complete = set(file_needed) < set(contents) and set(dir_needed) < set(contents)
                if complete:
                    target_data_dir = os.path.join(target_dir, env, params)
                    if os.path.exists(target_data_dir):
                        continue
                    os.makedirs(target_data_dir)

                    for file in file_needed:
                        shutil.copy(os.path.join(source_data_dir, file), os.path.join(target_data_dir, file))
                    for dir in dir_needed:
                        shutil.copytree(os.path.join(source_data_dir, dir), os.path.join(target_data_dir, dir))
                    print("finish {} {}".format(env, params))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=str)
    args = parser.parse_args()

    main(args.target_dir)
