from preprocess import *
from trainer import Classifier

if __name__ == '__main__':
    train, test = load_nsmc_set()
    sentences = convert_to_bert_input(train['document'])
    labels = extract_label(train['label'])

    bert_tokenizer = Tokenizer()
    tokenized_texts = bert_tokenizer.tokenize(sentences)
    token_idx = bert_tokenizer.token_to_idx(tokenized_texts)

    padding = padding_sentence(token_idx)
    attention_mask = load_attention_mask(padding)


    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(padding,
                                                                                        labels,
                                                                                        random_state=2018,
                                                                                        test_size=0.1)

    # 어텐션 마스크를 훈련셋과 검증셋으로 분리
    train_masks, validation_masks, _, _ = train_test_split(attention_mask,
                                                           padding,
                                                           random_state=2018,
                                                           test_size=0.1)
    # print(len(train_masks))
    # print(train_masks)
    bertclassifier = Classifier(train_inputs, train_labels, train_masks,
                                validation_inputs, validation_labels, validation_masks)
    bertclassifier.train()