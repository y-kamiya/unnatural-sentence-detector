import argparse
import random
import torch
from transformers import *
from logzero import setup_logger

class Detector:
    def __init__(self, config):
        self.config = config

        self.tokenizer, self.model = self.__create_model()

        with open(config.filepath, 'r') as f:
            self.sentences = [line.strip() for line in f.readlines()]
            if self.config.random:
                self.sentences = [self.__randomize(s) for s in self.sentences]

    def __create_model(self):
        if self.config.lang == 'ja':
            pretrained_weights = 'cl-tohoku/bert-base-japanese-whole-word-masking'
            tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_weights)
            model = BertForMaskedLM.from_pretrained(pretrained_weights)
            return tokenizer, model

        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        model = BertForMaskedLM.from_pretrained(pretrained_weights)
        return tokenizer, model

    def __randomize(self, sentence):
        words = sentence.split()
        centers = words[1:-1]
        random_words = [words[0]] + random.sample(centers, len(centers)) + [words[-1]]
        return ' '.join(random_words)

    def execute(self):
        for sentence in self.sentences:
            self.detect(sentence)

    def detect(self, sentence):
        list = []
        
        input_ids = torch.tensor([self.tokenizer.encode(sentence, add_special_tokens=True)])
        n_words = input_ids.shape[1]
        # replace each token except from <CLS>, <SEP> with <MASK>
        for i in range(1, n_words - 1):
            ids = input_ids.clone()
            ids[0][i] = self.tokenizer.mask_token_id
            list.append(ids)

        input = torch.cat(list, dim=0)

        with torch.no_grad():
            output = self.model(input)

        all_scores = output[0]
        print(all_scores.shape)

        logger = self.config.logger
        logger.debug('==========================')
        logger.debug(self.tokenizer.tokenize(sentence))
        is_strange = False
        total = 0
        for i in range(1, n_words - 1):
            scores = all_scores[i-1][i]
            topk = torch.topk(scores, 5)

            score = Score(input_ids[0][i], scores, self.tokenizer)
            top_scores = [Score(id.item(), scores, self.tokenizer) for id in topk.indices]
            is_strange = is_strange or score.value_std < self.config.threshold
            total += score.value_std

            logger.debug('original word: {}: top score: {}'.format(score, top_scores))

        average_score = total / (n_words - 2)
        result_text = 'ng' if is_strange else 'ok'
        result = '{}\t{:.3f}'.format(result_text, average_score)
        logger.info(result)

        if self.config.outputfile is not None:
            with open(self.config.outputfile, 'a') as f:
                f.write(result)
                f.write('\n')

class Score:
    def __init__(self, id, scores, tokenizer):
        self.id = id
        self.value = scores[id].item()
        self.word = tokenizer.decode([id])
        min_value = torch.min(scores)
        max_value = torch.max(scores)
        self.value_std = (self.value - min_value) / (max_value - min_value)

    def __repr__(self):
        return '({:.2f}, {})'.format(self.value_std, self.word)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('filepath', help='file path to target sentences')
    parser.add_argument('--lang', default='en', help='language')
    parser.add_argument('--threshold', type=float, default=0.8, help='sentence is strange when score is lower than this')
    parser.add_argument('--random', action='store_true', help='randomize word order')
    parser.add_argument('--outputfile', default=None)
    parser.add_argument('--loglevel', default='DEBUG')
    args = parser.parse_args()

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    detector = Detector(args)
    detector.execute()
