# import gendered pairs
filename_pairs  = 'penn-gender-pairs'
female_words, male_words =[],[]
with open(filename_pairs,'r') as f:
    gender_pairs = f.readlines()
f.close()

for gp in gender_pairs:
    f,m=gp.split()
    female_words.append(f)
    male_words.append(m)

gender_words = set(female_words) | set(male_words)

# read from "train.txt" and write to "trim_train.txt" with only lines containing gendered words
filename_r = 'data/penn/train.txt'
count = 0
with open(filename_r, 'r') as f:
    with open('trim_train.txt', 'w') as out:
        for line in f:
            words = line.split(' ')
            for word in words:
                if word in gender_words:
                    out.write(line)
                    # print(line)
                    count +=1
                    break
f.close()
out.close()
# print(count)
        