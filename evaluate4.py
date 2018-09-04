import os, sys, time

reload(sys)
sys.setdefaultencoding('utf-8')
from numpy import array
import numpy
import subprocess
import re
import nltk


# from nltk import translate


# nltk.set_proxy('http://s5.41115102.com:6124', ('chongyangtao@163.com', '2133366'))
# nltk.download('punkt')
# nltk.download('translate')


# retuen token list
def _split_into_words(text):
    # full_text_words = nltk.word_tokenize(text.lower().strip())
    full_text_words = text.lower().strip().split(' ')
    full_text_words = [w.replace('__', '') for w in full_text_words]
    return full_text_words


def _get_ngrams(n, text):
    words = _split_into_words(text)
    ngram_set = set()
    word_count = len(words)
    max_index_ngram_start = word_count - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(words[i:i + n]))
    return ngram_set


def rouge_n(evaluated, reference, n=2):
    """
    compuet ROUGE-N.
    :param evaluated: 
    :param reference: 
    :param n: length of n-grams
    :return float tuple: p , r , f_measure 
    """
    evaluated_ngrams = _get_ngrams(n, evaluated)
    reference_ngrams = _get_ngrams(n, reference)
    evaluated_count = len(evaluated_ngrams)
    reference_count = len(reference_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0 or reference_count == 0:
        p = 0.0
        r = 0.0
    else:
        p = float(overlapping_count) / float(evaluated_count)
        r = float(overlapping_count) / float(reference_count)

    if p and r:
        f_measure = 2.0 * r * p / (r + p)
    else:
        f_measure = 0.0

    return array([p, r, f_measure])


def _get_index_of_lcs(x, y):
    return len(x), len(y)


def _len_lcs(x, y):
    """
    Return the length of lcs betwwen x and y
    :param x: word order
    :param y: word order
    :returns integer: length of lcs bettwen x and y
    """
    table = _lcs(x, y)
    n, m = _get_index_of_lcs(x, y)
    return table[n, m]


def _lcs(x, y):
    """
    Return the lcs betwwen x and y
    citarion: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    :param x: 
    :param y: 
    :returns table: 
    """
    n, m = _get_index_of_lcs(x, y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _recon_lcs(x, y):
    """
    
    citation: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    :param x: 
    :param y: 
    :returns sequence: 
    """
    i, j = _get_index_of_lcs(x, y)
    table = _lcs(x, y)

    def _recon(i, j):
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
    return recon_tuple


def rouge_l(evaluated, reference):
    """
    Compuet ROUGE_L.
    :param evaluated: 
    :param reference: 
    :returns float tuple: p, r, f_measure
    """

    reference_words = _split_into_words(reference)
    evaluated_words = _split_into_words(evaluated)
    m = len(reference_words)
    n = len(evaluated_words)
    lcs = _len_lcs(evaluated_words, reference_words)
    if m!=0 or n!=0:
        p = float(lcs) / float(n)
        r = float(lcs) / float(m)
    else:
        p = 0.0
        r = 0.0

    if  p and r:
        f_measure = 2.0 * p * r / (r + p)
    else:
        f_measure = 0.0

    return array([p, r, f_measure])


def _get_skip_bigrams(text, k):
    """
    Get skip_bigrams from text
    :param text: 
    :param k: 
    :returns set: 
    """

    words = _split_into_words(text)
    skip_bigram_set = set()
    n = len(words)
    m = 0
    for w in words:
        m += 1
        i = m
        while (i - m <= k) & (i < n):
            skip_bigram_set.add((w, words[i]))
            i += 1
    return skip_bigram_set


def rouge_s(evaluated, reference, k):
    """
    Compuet ROUGE-S.
    :param evaluated: 
    :param reference: 
    :param k:
    :return float tuple: p, r, f_measure
    """
    skip_b_eval = _get_skip_bigrams(evaluated, k)
    skip_b_ref = _get_skip_bigrams(reference, k)

    overlap_skip_b = skip_b_eval.intersection(skip_b_ref)

    overlap_count = len(overlap_skip_b)
    eval_count = len(skip_b_eval)
    ref_count = len(skip_b_ref)
    if eval_count and ref_count:
        p = float(overlap_count) / float(eval_count)
        r = float(overlap_count) / float(ref_count)
    else:
        p = 0.0
        r = 0.0
    if p and r:
        f_measure = 2.0 * p * r / (r + p)
    else:
        f_measure = 0.0
    return array([p, r, f_measure])


# Compute location match ratio (equal length)
def lmr_(evaluated, reference):
    words_e = _split_into_words(evaluated)
    words_r = _split_into_words(reference)
    assert len(words_e) == len(words_r)
    match_count = 0
    for e, r in zip(words_e, words_r):
        if e == r:
            match_count = match_count + 1
    pmr = float(match_count) / float(len(words_e))
    return pmr


# Compute location match ratio
def lmr(evaluated, reference):
    words_e = _split_into_words(evaluated)
    words_r = _split_into_words(reference)

    match_count = 0
    for i in range(min(len(words_e), len(words_r))):
        if words_e[i] == words_r[i]:
            match_count = match_count + 1
    pmr = float(match_count) / float(len(words_r))
    return pmr


def bleu(evaluated, reference):
    words_e = _split_into_words(evaluated)
    words_r = _split_into_words(reference)
    bleu = translate.bleu_score.sentence_bleu([words_r], words_e)
    return bleu


def sys_bleu(result_txt, evaluated_list, reference_list):
    if not os.path.exists('tmp/'):
        os.mkdir('tmp/')
    evaluates = 'tmp/' + result_txt.split('.')[0] + '_e.txt'
    references = 'tmp/' + result_txt.split('.')[0] + '_r.txt'
    with open(evaluates, 'w') as f1:
        for item in evaluated_list:
            f1.writelines(item + '\n')
    with open(references, 'w') as f2:
        for item in reference_list:
            f2.writelines(item + '\n')

    ScoreBleu_path = 'zgen_bleu/ScoreBleu.sh'
    # command = "sh %s -t %s -r %s"%(ScoreBleu_path, evaluates, references)
    command = ["sh", ScoreBleu_path, "-t", evaluates, "-r", references]
    # os.system(command)
    process = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    result = out.decode()
    # print(result.strip())
    os.remove(evaluates)
    os.remove(references)
    sys_bleu_scores = re.findall(r"\= (.+?) \(", result)
    sys_bleu_score = float(sys_bleu_scores[0])
    return sys_bleu_score


def get_list(results):
    lines = open(results, 'r').readlines()
    evaluates = []
    references = []
    for i in range(0, len(lines), 7):
        # print(cmp(lines[i].strip(), lines[i+2].strip()))
        if len(lines[i + 1].strip().split(' ')) > 1500:
            continue
        if lines[i + 5].strip() == 'NULL':
            continue
        else:
            # print(lines[i+3].strip())
            # print(lines[i+5].strip())
            references.append(lines[i + 3].strip())
            evaluates.append(lines[i + 5].strip())
    '''
    for i, line in enumerate(lines):
        if (i+1)%3==2 and len(line.split(' '))<100:
           evaluates.append(line.strip()) 

           #print(lines.strip())
        if (i+1)%3==0 and len(line.split(' '))<100:
            references.append(line.strip())
            #print(lines.strip())
    '''
    return evaluates, references


def evaluate(result_txt, metrics="lmr"):
    evaluated_list, reference_list = get_list(result_txt)
    if metrics == 'pmr':
        count = 0
        for evaluated, reference in zip(evaluated_list, reference_list):
            if evaluated == reference:
                count = count + 1
        return float(count) / float(len(evaluated_list))

    if metrics == 'sys_bleu':
        return sys_bleu(result_txt, evaluated_list, reference_list)

    out = []
    for evaluated, reference in zip(evaluated_list, reference_list):
        if metrics == 'lmr':
            value = lmr(evaluated, reference)
        elif metrics == 'rouge_l':
            value = rouge_l(evaluated, reference)[0]
        elif metrics == 'bleu':
            value = bleu(evaluated, reference)
        # print(value)
        out.append(value)
    return numpy.mean(array(out))


def evaluate_all_matrics(result_txt_list, wf):
    data_name = '_'.join(result_txt_list[0].split('.')[0].split('_')[2:])
    wf.writelines('****************%s****************\n' % data_name)
    print('****************%s****************' % data_name)
    for metrics in ['rouge_l', 'pmr', 'lmr', 'bleu']:
        # for metrics in ['sys_bleu']:
        print('----------------%s----------------' % metrics)
        wf.writelines('----------------%s----------------\n' % metrics)
        for result_txt in result_txt_list:
            name = ' '.join(result_txt.split('.')[0].split('_')[1:])
            if os.path.exists(result_txt):
                result = evaluate(result_txt, metrics)
            else:
                result = 0.0
            print('%s of %s: %s' % (metrics, name, result))  # print("%.2f" % a)
            # wf.writelines(str(numpy.around(numpy.array(result), decimals=4)) + '\n')
            wf.writelines(("%.4f" % result) + '\n')


if __name__ == '__main__':
    base_fdir = 'results_eng'
    with open(os.path.join(base_fdir, 'eval.txt'), 'w') as wf:
        result_lstm = os.path.join(base_fdir, 'results_lstm.txt')
        result_order = os.path.join(base_fdir, 'results_order.txt')
        result_self = os.path.join(base_fdir, 'results_self.txt')
        result_self_mem = os.path.join(base_fdir, 'results_self_mem.txt')
        evaluate_all_matrics([result_lstm, result_order, result_self, result_self_mem], wf)
