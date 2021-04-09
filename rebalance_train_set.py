# A script to detect gendered words and replace them
# Run with
# `python rebalance_train_set.py [path to corpus file] [path to output file] [female %] [male %] --gender_pair_file [path of gender pair file]
# Example: python rebalance_train_set.py awd-lstm/data/penn/train.txt train_rebalanced.txt 50 50 --gender_pair_file penn-gender-pairs

import csv
import pandas as pd
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Output a rebalanced dataset')
parser.add_argument('corpus_path', help='Path to corpus text file', type=str)
parser.add_argument('output_path', help='Path to output text file', type=str)
parser.add_argument('male_percentage', help='Percentage of output male lines', type=int)
parser.add_argument('female_percentage', help='Percentage of output female lines', type=int)
parser.add_argument('--gender_pair_file', type=str, default='./penn-gender-pairs', help=('oath of gender-pairs file to be used'))
args = parser.parse_args()

# import dataset
test_set = pd.read_csv(args.corpus_path, delimiter="\n", names=['phrase'])
gender_pairs = pd.read_csv(args.gender_pair_file, delimiter=" ", names=['male', 'female'])

n = len(test_set)
count = np.array(range(n))

# detect female words
female = np.zeros(n, dtype=bool)
f_count = 0
for word in gender_pairs['female']:
    rgx = r'\b' + re.escape(word) + r'\b'
    idx = test_set['phrase'].str.contains(rgx, case=False, regex=True)
    f_count += sum(test_set['phrase'].str.count(rgx))
    female = np.logical_or(female, idx)

print('Number of female words:')
print(f_count)

# indices of lines that have female words
f_idx = count[np.array(female) == 1]

print('Number of lines with female words:')
print(len(f_idx))

# detect male words
male = np.zeros(n, dtype=bool)
m_count = 0

for word in gender_pairs['male']:
    rgx = r'\b' + re.escape(word) + r'\b'
    idx = test_set['phrase'].str.contains(rgx, case=False, regex=True)
    m_count += sum(test_set['phrase'].str.count(rgx))
    male = np.logical_or(male, idx)

print('Number of male words:')
print(m_count)

# indices of lines that have male words
m_idx = count[np.array(male) == 1]

print('Number of lines with male words:')
print(len(m_idx))

# indices of lines that have both male & female words
b_idx = count[np.logical_and(male, female)]
print('Number of lines with both gendered words:')
print(len(b_idx))

# Calculate number of lines that need replacements and their indices
total = len(f_idx) + len(m_idx) - len(b_idx)
female_goal = round(total / 100 * args.female_percentage)

print("Goal number of female sentences")
print(female_goal)
male_goal = round(total / 100 * args.male_percentage)

balance = int(female_goal - len(f_idx))

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
fr_count = 0

for word in gender_pairs['female']:
    rgx = r'\b' + re.escape(word) + r'\b'
    idx = updated_set['phrase'].str.contains(rgx, case=False, regex=True)
    fr_count += sum(updated_set['phrase'].str.count(rgx))
    female_rebalanced = np.logical_or(female_rebalanced, idx)

print('Number of female words:')
print(fr_count)
fr_idx = count[np.array(female_rebalanced) == 1]

print('Number of rebalanced lines with female gendered words:')
print(len(fr_idx))

# detect male words
male_rebalanced = np.zeros(n, dtype=bool)
mr_count = 0
for word in gender_pairs['male']:
    rgx = r'\b' + re.escape(word) + r'\b'
    idx = updated_set['phrase'].str.contains(rgx, case=False, regex=True)
    mr_count += sum(updated_set['phrase'].str.count(rgx))
    male_rebalanced = np.logical_or(male_rebalanced, idx)

print('Number of male words:')
print(mr_count)
mr_idx = count[np.array(male_rebalanced) == 1]

print('Number of rebalanced lines with male gendered words:')
print(len(mr_idx))

updated_set.to_csv(args.output_path, header=False, index=False)

