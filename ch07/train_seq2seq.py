import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq

(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char_to_id, id_to_char = sequence.get_vocab()

# Reverse input? =================================================
is_reverse = False
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
# ================================================================

vocab_size = len(char_to_id)
batch_size = 128
wordvec_size = 16
hidden_size = 128
max_epoch = 25
max_grad = 5.0

# Normal or Peeky? ==============================================
#model = Seq2seq(vocab_size, wordvec_size, hidden_size)
model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
# ================================================================
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)
    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i<10
        correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose)
    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc*100))
print(acc_list)

# グラフの描画
x = np.arange(len(acc_list))
baseline = np.array([0.18, 0.22, 0.56, 1.06, 2.28, 2.62, 2.34, 5.06, 5.38, 5.22,
                    3.44, 5.58, 7.22, 5.18, 7.06, 6.38, 9.64, 7.72, 8.66, 11.04,
                    10.08, 9.18, 4.94, 2.8, 10.78])/100
reverse = np.array([0.0012, 0.004, 0.0194, 0.0578, 0.1246, 0.1426, 0.175, 0.2308,
                    0.2654, 0.2982, 0.2848, 0.3664, 0.3942, 0.3664, 0.411, 0.427,
                    0.4284, 0.4064, 0.4754, 0.5002, 0.5098, 0.4772, 0.4522, 0.5184,
                    0.5408])
sota = np.array([0.0026, 0.0084, 0.02, 0.0552, 0.1136, 0.2884, 0.5708, 0.7562, 0.8586,
                0.8802, 0.932, 0.95, 0.9502, 0.964, 0.9646, 0.9686, 0.9688, 0.972,
                0.9784, 0.9306, 0.9814, 0.931, 0.9934, 0.9902, 0.9928])
plt.plot(x, baseline, color='blue', label='baseline')
plt.plot(x, reverse, color='green', label='reverse')
plt.plot(x, acc_list, color='yellow', label='peeky')
plt.plot(x, sota, color='red', label='reverse + peeky')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend()
plt.show()
