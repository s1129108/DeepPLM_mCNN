import h5py
import os

from tqdm import tqdm
from datetime import datetime

import numpy as np
import math

from sklearn import metrics
from sklearn.metrics import roc_curve

import tensorflow as tf
from tensorflow.keras import Model, layers

import gc
from sklearn.model_selection import KFold
"----------------------------------------------------------------------------------------------------"
import csv
import argparse
# How to use
# python MCNN_PLM.py -maxseq 1000 -f 256 -w 4 8 16 -nf 1024 -dt "A" -df "pt" -imb "None" -k 0 -csv "pred.csv"

parser = argparse.ArgumentParser(description='Program arguments')
parser.add_argument("-maxseq", "--MAXSEQ", type=int, default=1000)
parser.add_argument("-f", "--FILTER", type=int, default=256)
parser.add_argument("-w", "--WINDOW", nargs='+', type=int, default=[1, 2, 4, 5])
parser.add_argument("-nf", "--NUM_FEATURE", type=int)

parser.add_argument("-hi", "--HIDDEN", type=int, default=1000)
parser.add_argument("-drop", "--DROPOUT", type=float, default=0.7)
parser.add_argument("-ep", "--EPOCHS", type=int, default=20)

parser.add_argument("-dt", "--DATA_TYPE", type=str, default="A") # "A" or "B" (A:ionchannels, B:iontransporters)
parser.add_argument("-df", "--DATA_FEATURE", type=str, default="pt")
parser.add_argument("-imb", "--imbalance_mod", type=str, default="None", help="the mod for imbalance 'SMOTE','ADASYN','RANDOM'")
parser.add_argument("-csv", "--csv_path", type=str, default="MCNN_log.csv")

parser.add_argument("-k", "--KFold", type=int, default=0)
args = parser.parse_args()

MAXSEQ = args.MAXSEQ
NUM_FILTER = args.FILTER
WINDOW_SIZES = args.WINDOW
csv_file_path = args.csv_path

DATA_TYPE = args.DATA_TYPE
DATA_FEATURE = args.DATA_FEATURE

DROPOUT = args.DROPOUT
NUM_HIDDEN = args.HIDDEN

IMBALANCE = args.imbalance_mod

BATCH_SIZE  = 512


NUM_CLASSES = 2
CLASS_NAMES = ['Negative','Positive']


NUM_FEATURE = args.NUM_FEATURE
EPOCHS      = args.EPOCHS

K_Fold = args.KFold

if DATA_TYPE == "A":
    print(f"********************** pos : ionchannels_{DATA_FEATURE} ***************************")
elif DATA_TYPE == "B":
    print(f"********************** pos : iontransporters_{DATA_FEATURE} ***********************")


print("NUM_FILTER:", NUM_FILTER)
print("WINDOW_SIZES:", WINDOW_SIZES)
print("imb:", IMBALANCE)

"===================================================================================================="
def time_log(message):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(message, " : ", formatted_time)

"----------------------------------------------------------------------------------------------------"
# model
class DeepScan(Model):
	def __init__(self,
	             input_shape=(1, MAXSEQ, NUM_FEATURE),
	             window_sizes=[1024],
	             num_filters=256,
	             num_hidden=1000):
		super(DeepScan, self).__init__()
		# Add input layer
		self.input_layer = tf.keras.Input(input_shape)
		self.window_sizes = window_sizes
		self.conv2d = []
		self.maxpool = []
		self.flatten = []
		for window_size in self.window_sizes:
			self.conv2d.append(
			 layers.Conv2D(filters=num_filters,
			               kernel_size=(1, window_size),
			               activation=tf.nn.relu,
			               padding='valid',
			               bias_initializer=tf.constant_initializer(0.1),
			               kernel_initializer=tf.keras.initializers.GlorotUniform()))
			self.maxpool.append(
			 layers.MaxPooling2D(pool_size=(1, MAXSEQ - window_size + 1),
			                     strides=(1, MAXSEQ),
			                     padding='valid'))
			self.flatten.append(layers.Flatten())
		self.dropout = layers.Dropout(rate=DROPOUT)
		self.fc1 = layers.Dense(
		 num_hidden,
		 activation=tf.nn.relu,
		 bias_initializer=tf.constant_initializer(0.1),
		 kernel_initializer=tf.keras.initializers.GlorotUniform())
		self.fc2 = layers.Dense(NUM_CLASSES,
		                        activation='softmax',
		                        kernel_regularizer=tf.keras.regularizers.l2(1e-3))

		# Get output layer with `call` method
		self.out = self.call(self.input_layer)

	def call(self, x, training=False):
		_x = []
		for i in range(len(self.window_sizes)):
			x_conv = self.conv2d[i](x)
			x_maxp = self.maxpool[i](x_conv)
			x_flat = self.flatten[i](x_maxp)
			_x.append(x_flat)

		x = tf.concat(_x, 1)
		x = self.dropout(x, training=training)
		x = self.fc1(x)
		x = self.fc2(x)  #Best Threshold
		return x

"----------------------------------------------------------------------------------------------------"
# model fit batch funtion
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        return np.array(batch_data), np.array(batch_labels)
    
"----------------------------------------------------------------------------------------------------"
# predict
def get_predict(x_test, y_test):
    pred_test = model.predict(x_test[0:])
    fpr, tpr, thresholds = roc_curve(y_test[0:][:,1], pred_test[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='mCNN_MB')
    # display.plot()

    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print(f'Best Threshold={thresholds[ix]}, G-Mean={gmeans[ix]}')
    threshold = thresholds[ix]

    y_pred = (pred_test[:, 1] >= threshold).astype(int)

    TN, FP, FN, TP =  metrics.confusion_matrix(y_test[0:][:,1], y_pred).ravel()

    Sens = TP/(TP+FN) if TP+FN > 0 else 0.0
    Spec = TN/(FP+TN) if FP+TN > 0 else 0.0
    Acc = (TP+TN)/(TP+FP+TN+FN)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) if TP+FP > 0 and FP+TN > 0 and TP+FN and TN+FN else 0.0
    F1 = 2*TP/(2*TP+FP+FN)    
    return TP, FP, TN, FN, Sens, Spec, Acc, MCC, roc_auc, F1

"====================================== train data lodding ======================================"
#train_npz = np.load(f"get_feature/example/{DATA_FEATURE}_d{NUM_FEATURE}_L{MAXSEQ}/Class_{DATA_TYPE}_L{MAXSEQ}_d{NUM_FEATURE}_{IMBALANCE}_{DATA_FEATURE}.npz")
train_npz = np.load(f"get_feature/{DATA_FEATURE}_d{NUM_FEATURE}_L{MAXSEQ}/Class_{DATA_TYPE}_L{MAXSEQ}_d{NUM_FEATURE}_{IMBALANCE}_{DATA_FEATURE}.npz")

x_train = train_npz["feature"]
y_train = train_npz["label"]
print("Class:", DATA_TYPE, x_train.shape, y_train.shape)    


if K_Fold == 0:
    "====================================== Start independent ======================================"
    # shuffle
    num_samples = len(x_train)
    shuffle_indices = np.arange(num_samples)
    np.random.shuffle(shuffle_indices)

    # feature and label after shuffle
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    generator = DataGenerator(x_train, y_train, batch_size=BATCH_SIZE)

    "====================================== Model Train ======================================"
    time_log("Start Model Train")

    # 創建 TensorFlow 會話
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)


    model = DeepScan(
        num_filters=NUM_FILTER,
        num_hidden=NUM_HIDDEN,
        window_sizes=WINDOW_SIZES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=x_train.shape)  # Assuming x_train_A has the same shape as your input data

    model.summary()

    model.fit(generator, epochs=EPOCHS, shuffle=True)

    "====================================== Model save ======================================"
    base_path = "save_model/"

    model_name = f"Class_{DATA_TYPE}_L{MAXSEQ}_d{NUM_FEATURE}_{IMBALANCE}_{DATA_FEATURE}"

    # 檢查是否存在相同的文件名
    i = 0
    while os.path.exists(os.path.join(base_path, model_name + f"_{i}")):
        i += 1

    # 最終的文件名
    final_model_name = model_name + f"_{i}"

    # 保存模型
    model.save(os.path.join(base_path, final_model_name), save_format="tf")

    "====================================== predict ======================================"
    time_log("Start Model Test")
    "====================================== test data lodding ======================================"
    # test_npz = np.load(f"get_feature/example/{DATA_FEATURE}_d{NUM_FEATURE}_L{MAXSEQ}/Class_{DATA_TYPE}_L{MAXSEQ}_d{NUM_FEATURE}_test_{DATA_FEATURE}.npz")
    test_npz = np.load(f"get_feature/{DATA_FEATURE}_d{NUM_FEATURE}_L{MAXSEQ}/Class_{DATA_TYPE}_L{MAXSEQ}_d{NUM_FEATURE}_test_{DATA_FEATURE}.npz")

    x_test = test_npz["feature"]
    y_test = test_npz["label"]
    print("Class:", DATA_TYPE, x_test.shape, y_test.shape)
    TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1 = get_predict(x_test, y_test)
    
    Sens = f"{Sens:.4f}"
    Spec = f"{Spec:.4f}"
    Acc = f"{Acc:.4f}"
    MCC = f"{MCC:.4f}"
    AUC = f"{AUC:.4f}"
    
    print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens}, Spec={Spec}, Acc={Acc}, MCC={MCC}, AUC={AUC}, F1={F1}")

    # del model
    tf.compat.v1.reset_default_graph()
    session.close()

    # gc.collect()
    time_log("End Model Test")

else:
    "====================================== Start cross ======================================"
    time_log("Start cross")

    kfold = KFold(n_splits = K_Fold, shuffle = True, random_state = 2)
    results=[]
    i=1

    # 創建 TensorFlow 會話
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    for train_index, test_index in kfold.split(x_train):    
        print(i,"/",K_Fold,'\n')
        # 取得訓練和測試數據
        X_train, X_test = x_train[train_index], x_train[test_index]
        Y_train, Y_test = y_train[train_index], y_train[test_index]
        print(X_train.shape)
        print(X_test.shape)
        print(Y_train.shape)
        print(Y_test.shape)


        time_log("Start Model Train")

        # 重新建模
        model = DeepScan(
        num_filters=NUM_FILTER,
            num_hidden=NUM_HIDDEN,
            window_sizes=WINDOW_SIZES)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.build(input_shape=X_train.shape)
        model.summary()
        # 在測試數據上評估模型
        model.fit(
            X_train,
            Y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)],
            verbose=1
        )
        time_log("End Model Train")

        i+=1
        
        time_log("Start Model Test")
        TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1 = get_predict(X_test, Y_test)
        results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC])
    
        del X_train
        del X_test
        del Y_train
        del Y_test

        model.reset_states()
        del model

        gc.collect()
        time_log("End Model Test")
    
        # 釋放 GPU 記憶體
    tf.compat.v1.reset_default_graph()
    session.close()
    # tf.keras.backend.clear_session()
    time_log("End cross")

    "====================================== cross predict ======================================"
    mean_results = np.mean(results, axis=0)

    TP = f"{mean_results[0]:.4}"
    FP = f"{mean_results[1]:.4}"
    TN = f"{mean_results[2]:.4}"
    FN = f"{mean_results[3]:.4}"
    Sens = f"{mean_results[4]:.4}"
    Spec = f"{mean_results[5]:.4}"
    Acc = f"{mean_results[6]:.4}"
    MCC = f"{mean_results[7]:.4}"
    AUC = f"{mean_results[8]:.4}"

    print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens}, Spec={Spec}, Acc={Acc}, MCC={MCC}, AUC={AUC}")

"====================================== result to CSV ======================================"
result_dict = [
    ["TestSet", "Feature", "TP", "FP", "TN", "FN", "Filter", "seq length", "windows", "Sens", "Spec", "Acc", "MCC", "AUC", "imb", "hidden", "Dropout", "epochs", "Num"],
    [DATA_TYPE, DATA_FEATURE, TP, FP, TN, FN, NUM_FILTER, MAXSEQ, str(WINDOW_SIZES)[1:-1], Sens, Spec, Acc, MCC, AUC, IMBALANCE, NUM_HIDDEN, DROPOUT, EPOCHS, i]
]    


# 檢查檔案是否存在，若存在則附加資料，否則建立新檔案並寫入資料
mode = 'a' if os.path.exists(csv_file_path) else 'w'
with open(csv_file_path, mode, newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    if mode == 'w':
        writer.writerow(result_dict[0])  # 寫入欄位名稱
    writer.writerows(result_dict[1:])  # 寫入資料

print(f'CSV file written successfully!')