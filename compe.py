#win環境ではcsvファイルの文字コードはANSIである必要がある．

import tensorflow as tf
import numpy as np
import csv

'''パラメタ'''
datanum = 11
labnum = 5
step_num = 10000
num_units = 25
batch_size = 111
'''csvファイルからデータの読み込み'''
#csvdataのかたちは以下のように
# 5.2, 6.1, 7.2, 8.2, 1(int)
# 2.3, 4.5, 6.7, 2.4, 2(int)
# ...
# data,data,data,data,label

def csv_loader(filename):
    csv_obj = csv.reader(open(filename, "r"))
    dt = [ v for v in csv_obj]
    dat = [[float(elm) for elm in v] for v in dt]
    db = [[0 for n in range(datanum)] for m in range(len(dat))]
    lb = [0 for mm in range(len(dat))]
    for i in range(len(dat)):
        for j in range(len(dat[i])):
            if j <= datanum - 1:
                db[i][j] = dat[i][j]
            else:
                lb[i] = int(dat[i][j])-4
    #lbをone_hot表現にする
    #lb_onehot = np.identity(labnum)[lb]
    return (db,lb)

#csvデータをtrainとtestで別々に格納
data_body,label_dum = csv_loader("train.csv")
data_test_body,label_test_dum = csv_loader("test.csv")
label_body = np.identity(labnum)[label_dum]
label_test_body = np.identity(labnum)[label_test_dum]
#ndarrayに変換
data_body = np.array(data_body)
data_test_body = np.array(data_test_body)

def zscore(x):
    xmean = x.mean()
    xstd  = np.std(x)

    zscore = (x-xmean)/xstd
    return zscore

data_body = zscore(data_body)
data_test_body = zscore(data_test_body)

#placeholder
data = tf.placeholder(dtype=tf.float32,shape=[None,datanum])
label = tf.placeholder(dtype=tf.float32,shape=[None,labnum])
keep_prob = tf.placeholder(tf.float32)

#隠れ層
b3 = tf.Variable(tf.zeros([num_units]))
w3 = tf.Variable(tf.truncated_normal([datanum, num_units]))
hidden3 = tf.nn.relu(tf.matmul(data,w3) + b3)
drop_out_h3 = tf.nn.dropout(hidden3, keep_prob)

b2 = tf.Variable(tf.zeros([num_units]))
w2 = tf.Variable(tf.truncated_normal([num_units, num_units]))
hidden2 = tf.nn.relu(tf.matmul(drop_out_h3,w2) + b2)
drop_out_h2 = tf.nn.dropout(hidden2, keep_prob)

b1 = tf.Variable(tf.zeros([num_units]))
w1 = tf.Variable(tf.truncated_normal([num_units, num_units]))
hidden1 = tf.nn.relu(tf.matmul(drop_out_h2,w1) + b1)
drop_out_h1 = tf.nn.dropout(hidden1, keep_prob)

b0 = tf.Variable(tf.zeros([labnum],dtype=tf.float32))
w0 = tf.Variable(tf.zeros([num_units,labnum],dtype=tf.float32))
y = tf.nn.softmax(tf.matmul(drop_out_h1,w0) + b0)

#単層の場合はこちらを使う
#w0 = tf.Variable(tf.zeros([datanum,labnum],dtype=tf.float32))
#y = tf.nn.softmax(tf.matmul(data,w0) + b0)


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=y))

# Regularization terms (weight decay)
#L2_sqr = (tf.nn.l2_loss(w3)
#          + tf.nn.l2_loss(w2)
#          + tf.nn.l2_loss(w1))
#lambda_2 = 0.01

#loss = cross_entropy + lambda_2 * L2_sqr

def training(loss):
    with tf.name_scope('training') as scope:
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    return train_step

#train_step = training(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.004).minimize(cross_entropy)
#train_step = tf.train.AdagradOptimizer(0.01).minimize(loss)
#train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    #ミニバッチ
    for i in range(step_num):
        sff_idx = np.random.permutation(data_body.shape[0])
        for idx in range(0, data_body.shape[0], batch_size):
            batch_x = data_body[sff_idx[idx: idx+batch_size]]
            batch_l = label_body[sff_idx[idx: idx+batch_size]]
            s.run(train_step,feed_dict={data:batch_x,label:batch_l,keep_prob:1.0})

    acc = s.run(accuracy, feed_dict={data:data_test_body,label:label_test_body,keep_prob:1.0})
    print("結果：{:.2f}%".format(acc * 100))
    y_vals = s.run(y, feed_dict={data:data_test_body,label:label_test_body,keep_prob:1.0})
    #print(y_vals)
    #y_value = np.argmax(y_vals,axis = 1)+4
    y_value = []
    for i in y_vals:
        sum = 4
        for j in range(labnum):
            sum += j*i[j]
        y_value.append(sum)

    print(y_value)
    np.savetxt('out.csv',y_value,delimiter='\n,')
    #w_vals = s.run(w0, feed_dict={data:data_test_body,label:label_test_body})
    #print(w_vals)
