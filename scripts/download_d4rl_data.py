import argparse
import os
import pickle
import gym
import d4rl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--dataset_name", type=str, default="hopper-medium-expert-v2")
    arglist = parser.parse_args()

    arglist = vars(parser.parse_args())
    return arglist

def main(arglist):
    dataset_name = arglist["dataset_name"]
    data_path = arglist["data_path"]
    save_path = os.path.join(data_path, "d4rl")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    env = gym.make(dataset_name)
    dataset = env.get_dataset()
    
    with open(os.path.join(save_path, f"{dataset_name}.p"), "wb") as f:
        pickle.dump(dataset, f)
        
    print("dataset saved at: {}".format(os.path.join(save_path, f"{dataset_name}.p")))

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)