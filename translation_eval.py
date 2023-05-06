# import evaluate
import os
import json
# from nltk.translate.bleu_score import sentence_bleu

# pred_dir = '/mnt/user/E-zhaoyingxiu.zyx-354256/MODELS/chavinlo-alpaca-native'
pred_dir = './'
# pred_file = os.path.join(pred_dir,'pred_trans.json')
pred_file = os.path.join(pred_dir,'epoch4_pred.json')
# ref_file = '/mnt/user/E-zhaoyingxiu.zyx-354256/LLMDATA/wmt_test.json'
ref_file = './wmt_test.json'

# refout = './ref.txt'
predout = './epoch4_pred.txt'
# predout = './backbone_pred.txt'

with open(pred_file) as f:
    res_list = [json.loads(i) for i in f.readlines()]
    predictions = [i['pred'].replace('\n', '') for i in res_list]
print(len(predictions))

with open(predout, 'w') as f:
    for i in predictions:
        print(i, file=f)

# references = []
# with open(ref_file) as f:
#     ref_list = json.load(f)
#     for sample in ref_list:
#         references.append(sample['output'])

# with open(refout, 'w') as f:
#     for i in references:
#         print(i, file=f)

# num = 100
# print(f'Begin load bleu...')
# bleu = evaluate.load("bleu")
# print(f'Load bleu done.')
# results = bleu.compute(predictions=predictions[:num], references=references[:num])
# print(f'Final results: {results}', flush=True)
# print(f"BLEU score: {results['bleu']}", flush=True)
# res = []
# for i in range(100):
#     pred = predictions[i].split()
#     print(pred)
#     ref = [references[i].split()]
#     print(ref)
#     bleu = sentence_bleu(ref, pred)
#     res.append(bleu)
# print(res)
# print(sum(res)/len(res))


# print(f'Congrats!')
