import tensorflow as tf
import torch
import numpy as np
import datetime
import time
import random

from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from data_loader import convert_to_tensor

# 정확도 계산 함수
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# 시간 표시 함수
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

class Classifier:
    def __init__(self, train_inputs, train_labels, train_masks,
                 eval_inputs, eval_labels, eval_masks):
        # 분류를 위한 BERT 모델 생성
        self.device = torch.device("cpu")
        self.model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
        self.model.to(self.device)

        self.train_dataloader = convert_to_tensor(train_inputs, train_labels, train_masks)
        self.validation_dataloader = convert_to_tensor(eval_inputs, eval_labels, eval_masks)

        self.optimizer = AdamW(self.model.parameters(),
                               lr=2e-5,  # 학습률
                               eps=1e-8  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                               )
        self.epochs = 4
        self.total_steps = len(self.train_dataloader) * self.epochs # 총 훈련 스텝 : 배치반복 횟수 * 에폭
        self.train_scheduler()

    def train_scheduler(self):

        # 처음에 학습률을 조금씩 변화시키는 스케줄러 생성
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         num_training_steps=self.total_steps)

    def train(self):
        # 재현을 위해 랜덤시드 고정
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # 그래디언트 초기화
        self.model.zero_grad()

        # 에폭만큼 반복
        for epoch_i in range(0, self.epochs):

            # ========================================
            #               Training
            # ========================================

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # 시작 시간 설정
            t0 = time.time()

            # 로스 초기화
            total_loss = 0

            # 훈련모드로 변경
            self.model.train()

            # 데이터로더에서 배치만큼 반복하여 가져옴
            for step, batch in enumerate(self.train_dataloader):

                # 경과 정보 표시
                if step % 500 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))

                # 배치를 GPU에 넣음
                batch = tuple(t.to(self.device) for t in batch)

                # 배치에서 데이터 추출
                b_input_ids, b_input_mask, b_labels = batch

                b_input_ids = b_input_ids.to(self.device)
                b_input_mask = b_input_mask.to(self.device)
                b_labels = b_labels.to(self.device)
                # Forward 수행
                outputs = self.model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)
                print(outputs)
                # 로스 구함
                loss = outputs[0]

                # 총 로스 계산
                total_loss += loss.item()

                # Backward 수행으로 그래디언트 계산
                loss.backward()

                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # 그래디언트를 통해 가중치 파라미터 업데이트
                self.optimizer.step()

                # 스케줄러로 학습률 감소
                self.scheduler.step()

                # 그래디언트 초기화
                self.model.zero_grad()

            # 평균 로스 계산
            avg_train_loss = total_loss / len(self.train_dataloader)
            torch.save(self.model.load_state_dict(), './model')
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            # 시작 시간 설정
            t0 = time.time()

            # 평가모드로 변경
            self.model.eval()

            # 변수 초기화
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # 데이터로더에서 배치만큼 반복하여 가져옴
            for batch in self.validation_dataloader:
                # 배치를 GPU에 넣음
                batch = tuple(t.to(self.device) for t in batch)

                # 배치에서 데이터 추출
                b_input_ids, b_input_mask, b_labels = batch

                # 그래디언트 계산 안함
                with torch.no_grad():
                    # Forward 수행
                    outputs = self.model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)

                # 로스 구함
                logits = outputs[0]

                # CPU로 데이터 이동
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # 출력 로짓과 라벨을 비교하여 정확도 계산
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print("  Validation took: {:}".format(format_time(time.time() - t0)))

        print("")
        print("Training complete!")
