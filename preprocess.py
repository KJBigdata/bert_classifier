import pandas as pd
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from data_loader import load_nsmc_set

def convert_to_bert_input(sentences: [pd.DataFrame, list]):
    sentences = ["[CLS] " + str(sentence) + " [SEP]"
                 for sentence in sentences]

    return sentences

def extract_label(labels: pd.DataFrame):
    return labels.values

def padding_sentence(input_idx, sequence_max_len = 128):
    return pad_sequences(input_idx, maxlen=sequence_max_len,
                         dtype="long", truncating="post", padding="post")

def load_attention_mask(padding):
    attention_masks = []
    for seq in padding:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    return attention_masks

def split_data(input_ids, labels, random_state=2018, test_size=0.1):
    return train_test_split(input_ids, labels, random_state=2018, test_size=0.1)

class Tokenizer:

    def __init__(self):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        self.max_len = 128 #입력 토큰의 최대 시퀀스 길이

    def tokenize(self, sentences: list):

        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_texts

    def token_to_idx(self, tokenized_texts):
        input_idx = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

        return input_idx



