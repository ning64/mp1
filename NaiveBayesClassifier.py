#coding:utf-8
import sys
import time
import math
import random
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')
def compute_acc(list1,list2):
    if len(list1)!=len(list2):
        print 'len not equal'
    else:
        nt=len(list1)
        nc=0
        for x in range(0,len(list1)):
            if list1[x]==list2[x]:
                nc+=1
        return (float(nc)/nt)

def divide_training_set(seq, num):
    random.Random(0).shuffle(seq)
    avg = len(seq) / float(num)
    output = []
    last = 0.0
    while last < len(seq):
        output.append(seq[int(last):int(last + avg)])
        last += avg
    return output

def cross_validation(para_list,train_set):
    score_board=[]
    for para in para_list:
        para_score=[]
        for i in range(0,len(train_set)):
            train_group=[]
            for m in range(0,len(train_set)):
                if m!=i:
                    train_group+=train_set[m]
                else:
                    val_group=train_set[m]
            (word_dict, author_list) = train_bayes(train_group, para)
            (acc_score, prediction, test_labels) = test_model(val_group, word_dict, author_list)
            para_score.append(acc_score)
        score_board.append(np.mean(para_score))
    para_f=para_list[score_board.index(max(score_board))]
    print 'score_board'
    print score_board
    return para_f






def train_bayes(train_data,fract,alpha=0.0000001):
    #v= vocab, doc=document, a=author, dv=distinct word
    #grab general info of the word
    v_word_dict = {}
    dv_tot = 0
    doc_tot=0
    v_tot=0


    #grab the general info of the author
    a_list = [0] * 15

    #grab the relation between word and author
    a_v_word_dict={}

    #final_out put
    f_word_dict = {}
    f_a_list=[0] * 15


    for i in range(0,15):
        a_v_word_dict[i]={}
        f_word_dict[i]={}
    'train data'
    for data in train_data:
        doc = data[0]
        label = int(data[1])
        doc_tot += 1
        a_list[(label - 1)] += 1

        for word in doc:
            v_tot += 1
            if v_word_dict.has_key(word) == False:
                v_word_dict[word] = 0
                dv_tot += 1
            if a_v_word_dict[(label - 1)].has_key(word) == False:
                a_v_word_dict[(label - 1)][word] = 0

            v_word_dict[word] += 1
            a_v_word_dict[(label - 1)][word] += 1

    'cal P(word/author) probability'
    for word in v_word_dict.keys():
        nkv=v_word_dict[word]
        for i in range(0, 15):
            if a_v_word_dict[i].has_key(word):
                nkva = a_v_word_dict[i][word]
            else:
                nkva = 0
            rv_score=-1/(math.log(((float(nkva) + alpha) / (v_word_dict[word] + alpha * dv_tot))))
            IDFv_score=float(v_tot)/nkv
            f_word_dict[i][word] = (fract)*IDFv_score*rv_score/(a_list[i])+(1-fract)*IDFv_score*rv_score
            
    'cal P(author)'
    for i in range(0, 15):
        f_a_list[i] = -1/math.log(float(a_list[i]) / doc_tot)
    'train_end'
    return (f_word_dict,f_a_list)

def test_model(test_data,word_dict,author_list):
    predictions = []
    test_labels = []
    n=0
    for data in test_data:
        n+=1
        doc = data[0]
        label = int(data[1])
        score = author_list[:]
        for word in doc:
            for i in range(0, 15):
                score[i] = score[i] + word_dict[i][word]
        prediction=score.index(max(score)) + 1
        predictions.append(prediction)
        test_labels.append(label)

    acc_score = compute_acc(predictions, test_labels)
    return (acc_score,predictions,test_labels)





def read_input(lines):
    data=[]
    for line in lines:
        line=line.strip()
        if len(line.split(','))==2:
            article=line.split(',')[0]
            label=line.split(',')[1]
            data.append([article.split(' ')[:-1],label])
    return data

if __name__ == '__main__':
    'read data'
    train_start=time.time()
    f1=open(sys.argv[1],'r')
    f2=open(sys.argv[2],'r')

    train_data = read_input(f1)
    test_data = read_input(f2)

    #para_list=[0.0061,0.0062,0.0063]
    #train_set=divide_training_set(train_data, 3)
    #para_f=cross_validation(para_list, train_set)

    (word_dict, author_list)=train_bayes(train_data,0.9938)
    train_end=time.time()

    test_start=time.time()
    (acc_score_t, predictions_t, test_labels_t) = test_model(train_data, word_dict, author_list)
    (acc_score, predictions, test_labels)=test_model(test_data, word_dict, author_list)
    test_end=time.time()
    for prediction in predictions:
        print prediction
    print '%d,%s' % (int(train_end-train_start),'seconds (training)')
    print '%d,%s' % (int(test_end-test_start),'seconds (testing)')
    print '%.3f,%s' % (acc_score_t, '(training)')
    print '%.3f,%s' % (acc_score, '(testing)')








