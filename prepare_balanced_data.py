import json, os, sys
import random

# numdata = 10000
numdata=5000

datadir = '../../LLMDATA/'
outdir = '../../LLMDATA/'

file1 = os.path.join(datadir, 'alpaca_data_cleaned.json')
file2 = os.path.join(datadir, 'wmt19_en-zh.json')

outfile = os.path.join(outdir, str(numdata)+'_mixed_inst_trans.json')

with open(outfile, 'w') as fw:
    with open(file1, 'r') as f1:
        data = json.load(f1)
    instruction_data = random.choices(data, k=numdata)
    with open(file2, 'r') as f2:
        data = json.load(f2)
    translation_data = random.choices(data, k=numdata)
    alldata = instruction_data + translation_data
    print(f'Total used data number is {len(alldata)}', flush=True)
    random.shuffle(alldata)
    json.dump(alldata, fw, ensure_ascii=False, indent=4)

print(f'Congrats!!!', flush=True)

