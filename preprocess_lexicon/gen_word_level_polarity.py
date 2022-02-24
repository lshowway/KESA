# generate word level sentiment polarity from SWN 3.0
from collections import defaultdict
from ast import literal_eval


def read_write_SWN():
    r_file = "D:/nutstore\phd4\AAAI_SA\dataset\lexicon/SentiWordNet_3.0.0.txt"


    word_sentiment_dic = defaultdict(dict)
    with open(r_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            POS, ID, PosScore, NegScore, SynsetTerms, Gloss = line
            PosScore, NegScore = float(PosScore), float(NegScore)
            words = SynsetTerms.split()
            for word in words:
                word, num = word.split('#')
                word_sentiment_dic[word].update({num: PosScore - NegScore})

    return word_sentiment_dic


def extract_word_polarity(word_sentiment_dic):
    # 抽取每个word的最强极性
    word_polarity = {}
    for word, polarity_list in word_sentiment_dic.items():
        values = list(polarity_list.values())
        max_value = max(values)
        min_value = abs(min(values))
        if max_value > 0 and max_value > min_value:
            word_polarity[word] = 0
        if min_value > 0 and min_value > max_value:
            word_polarity[word] = 1

    # fw = open('../../datasets/lexicon/SWN.word.polarity', 'w', encoding='utf-8')
    # fw.write(str(word_polarity))





if __name__ == "__main__":
    t1 = read_write_SWN()
    extract_word_polarity(t1)