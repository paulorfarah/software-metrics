import argparse

import pandas as pd


def main(commits_file):
    print('starting...')
    file = 'C:\\Users\\paulo\\ufpr\\datasets\\software-metrics\\resources\\resources-csv-1.csv'
    commits_list = ['1bd1fd8e6065da9d07b5a3a1723b059246b14001', 'e8f24e86bb2d54493e3f0c0bd7787abb1d1d7443',
                    '1914e7daae2cb39451046e67b993c8ab77e34397']

    # with open(args.commits) as f:
    #     commits_list = f.read().splitlines()
    #     commits_list.reverse()
    df = pd.read_csv(file)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='evaluate resources')
    ap.add_argument('--commits', required=False, help='csv with a list of commits (newest to oldest) to compare commitA and commitB')
    args = ap.parse_args()
    main(args.commits)