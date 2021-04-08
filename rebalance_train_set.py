# A script to detect gendered words and replace them
# Run with
# `python rebalance_train_set.py [path to corpus file] [path to output file] --gender_pair_file [path of gender pair file]
# Example: python rebalance_train_set.py awd-lstm/data/penn/train.txt train_rebalanced.txt --gender_pair_file penn-gender-pairs

import csv
import pandas as pd
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Output the bias scores of a corpus')
parser.add_argument('corpus_path', help='Path to corpus text file', type=str)
parser.add_argument('output_path', help='Path to output text file', type=str)
parser.add_argument('--gender_pair_file', type=str, default='./penn-gender-pairs', help=('oath of gender-pairs file to be used'))
args = parser.parse_args()

# import dataset
test_set = pd.read_csv(args.corpus_path, delimiter="\n", names=['phrase'])
gender_pairs = pd.read_csv(args.gender_pair_file, delimiter=" ", names=['male', 'female'])

n = len(test_set)
count = np.array(range(n))

# detect female words
female = np.zeros(n, dtype=bool)

for word in gender_pairs['female']:
    rgx = r'\b' + re.escape(word) + r'\b'
    idx = test_set['phrase'].str.contains(rgx, case=False, regex=True)
    female = np.logical_or(female, idx)

# indices of lines that have female words
f_idx = count[np.array(female) == 1]

print('Number of lines with female words:')
print(len(f_idx))

# detect male words
male = np.zeros(n, dtype=bool)

for word in gender_pairs['male']:
    rgx = r'\b' + re.escape(word) + r'\b'
    idx = test_set['phrase'].str.contains(rgx, case=False, regex=True)
    male = np.logical_or(male, idx)

# indices of lines that have male words
m_idx = count[np.array(male) == 1]
print('Number of lines with male words:')
print(len(m_idx))

# indices of lines that have both male & female words
b_idx = count[np.logical_and(male, female)]
print('Number of lines with both gendered words:')
print(len(b_idx))

# Calculate number of lines that need replacements and their indices
balance = int((len(m_idx) - len(b_idx) - len(f_idx)) / 2)
select_idx = np.array([s for s in m_idx if s not in b_idx])

# Switch from male to female: all lines that has both male/female and some extra male lines
switch_idx = np.random.choice(select_idx, size=balance, replace=False)
switch_idx = np.append(switch_idx, b_idx)

# create a copy to be the updated set
updated_set = test_set.copy()

# update values
print('Rebalancing...')
for idx in switch_idx:
    for g_idx in range(len(gender_pairs)):
        rgx = r'\b' + re.escape(gender_pairs['male'][g_idx]) + r'\b'
        updated_set.iloc[idx, 0] = re.sub(rgx, gender_pairs['female'][g_idx], updated_set.iloc[idx, 0])

# recalculate number of female lines
female_rebalanced = np.zeros(n, dtype=bool)

for word in gender_pairs['female']:
    rgx = r'\b' + re.escape(word) + r'\b'
    idx = updated_set['phrase'].str.contains(rgx, case=False, regex=True)
    female_rebalanced = np.logical_or(female_rebalanced, idx)

fr_idx = count[np.array(female_rebalanced) == 1]

print('Number of rebalanced lines with female gendered words:')
print(len(fr_idx))

# detect male words
male_rebalanced = np.zeros(n, dtype=bool)

for word in gender_pairs['male']:
    rgx = r'\b' + re.escape(word) + r'\b'
    idx = updated_set['phrase'].str.contains(rgx, case=False, regex=True)
    male_rebalanced = np.logical_or(male_rebalanced, idx)

mr_idx = count[np.array(male_rebalanced) == 1]

print('Number of rebalanced lines with male gendered words:')
print(len(mr_idx))

updated_set.to_csv(args.output_path, header=False, index=False)