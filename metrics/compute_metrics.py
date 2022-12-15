import argparse
import nltk.translate.gleu_score as gleu
from nltk.translate.bleu_score import sentence_bleu
import nltk.translate.chrf_score as chrf
from nlgeval import compute_metrics
import pandas as pd
from rouge import FilesRouge
from rouge_score import rouge_scorer, scoring
import numpy as np
import subprocess
import os, time, json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu


def score_gleu(reference, hypothesis):
    score = 0
    for ref, hyp in zip(reference, hypothesis):
        score += gleu.sentence_gleu([ref.split()], hyp.split())
    return float(score) / len(reference)


def get_rouge(hyp_path, ref_path):
    files_rouge = FilesRouge()
    scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
    return scores


def compute(preds, gloden):
    t = open(gloden, 'r', encoding='utf8')
    p = open(preds, 'r', encoding='utf8')
    tline = t.readlines()
    pline = p.readlines()
    gleu_result = score_gleu(tline, pline)
    print('GLEU : ', gleu_result)

    metrics_dict = compute_metrics(hypothesis=preds,
                                   references=[gloden], no_skipthoughts=True, no_glove=True)

    return gleu_result


def computeBleu1_to_4(reference_list, candidate_list, filename, i):
    bleu1_sum = bleu2_sum = bleu3_sum = bleu4_sum = bleuA_sum = 0

    for (ref, cand) in zip(reference_list, candidate_list):

        tokens_real = ref.split(' ')
        tokens_pred = cand.split(' ')

        if cand == '':
            bleu1_score = bleu2_score = bleu3_score = bleu4_score = bleuA_score = 0

        else:
            bleu1_score = sentence_bleu([tokens_real], tokens_pred, weights=(1.0, 0.0, 0.0, 0.0))
            bleu2_score = sentence_bleu([tokens_real], tokens_pred, weights=(0.0, 1.0, 0.0, 0.0))
            bleu3_score = sentence_bleu([tokens_real], tokens_pred, weights=(0.0, 0.0, 1.0, 0.0))
            bleu4_score = sentence_bleu([tokens_real], tokens_pred, weights=(0.0, 0.0, 0.0, 1.0))
            bleuA_score = sentence_bleu([tokens_real], tokens_pred, weights=(0.25, 0.25, 0.25, 0.25))

        bleu1_sum += bleu1_score
        bleu2_sum += bleu2_score
        bleu3_sum += bleu3_score
        bleu4_sum += bleu4_score
        bleuA_sum += bleuA_score

    output = 'BLEU_[A-1-2-3-4]: {}/{}/{}/{}/{}'.format(
        round(bleuA_sum / len(reference_list), 3) * 100,
        round(bleu1_sum / len(reference_list), 3) * 100,
        round(bleu2_sum / len(reference_list), 3) * 100,
        round(bleu3_sum / len(reference_list), 3) * 100,
        round(bleu4_sum / len(reference_list), 3) * 100
    )

    filename.write('[TOKENS {}]: {}\n'.format(i, output))


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )

    return matrix[size_x - 1, size_y - 1]


def computePerfectPrediction(predictions_file, target_file):
    perfect_pred = {}

    target = open(target_file, 'r')
    target_list = target.readlines()

    for i in range(1, 16):

        perfect_pred[i] = 0

        with open(predictions_file) as fread:

            data = fread.readlines()

            for (ref, pred) in zip(target_list, data):

                pred = pred.strip().lower().split()
                pred = pred[0:i]

                ref = ref.strip().lower().split()
                ref = ref[0:i]

                if len(ref) >= i and len(pred) >= i:
                    if ''.join(pred) == ''.join(ref):
                        perfect_pred[i] += 1

            len_tokens_subset = len([item.strip() for item in target_list if len(item.split()) >= i])
            perfect_pred[i] = (perfect_pred[i], len_tokens_subset)

    return perfect_pred


def levenshteinOverList(predictions_file, references_file):
    p = open(predictions_file, 'r')
    data = p.readlines()

    r = open(references_file, 'r')
    target_list = r.readlines()

    references = {}
    candidates = {}
    lev_list = {}

    for i in range(1, 16):
        references[i] = []
        candidates[i] = []
        lev_list[i] = 0

    for i in range(1, 16):

        for (ref, pred) in zip(target_list, data):

            target_lev = len(ref.strip().split())

            if target_lev >= i:
                pred = pred.strip().lower().split()[0:i]
                ref = ref.strip().lower().split()[0:i]
                references[i].append(ref)
                candidates[i].append(candidates)
                lev_list[i] += levenshtein(ref, pred)

    for i in [1, 3, 5]:
        lev = lev_list[i]
        output = 'LEV: {}'.format(round(lev / len(references[i]), 3))
        print('[TOKENS {}]: {}'.format(i, output))


if __name__ == "__main__":
    lans = ['python', 'java', 'c#', 'javascript', 'php', 'ruby', 'go', 'html']
    path = "../results/"
    for lan in lans:
        print(lan)
        pre_path = path + 'predict_' + lan + '.csv'
        gol_path = path + 'Golden_' + lan + '.csv'

        compute(pre_path, gol_path)
        rouge_list = get_rouge(pre_path, gol_path)

        perfect_score = computePerfectPrediction(pre_path, gol_path)
        perfect_predictions = {}

        for i in [1, 3, 5]:
            numbers, len_data = perfect_score[i]
            percentage = (numbers / len_data) * 100
            print('[@{} tokens]: Perfect {}: {}% ({}/{})'.format(i, lan, percentage, numbers, len_data))
            perfect_predictions[i] = {'percentage': percentage, '#perfect': numbers, '#tokens': len_data}

        references_overall = []
        candidates_overall = []

        with open(pre_path, 'r') as fread:
            for item in fread.readlines():
                candidates_overall.append(item.strip().lower())

        with open(gol_path, 'r') as fread:
            for item in fread.readlines():
                references_overall.append(item.strip().lower())

        levenshteinOverList(pre_path, gol_path)
        print('\n')