# Import BeautifulSoup
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import time, os, random, bs4
from transformers import AdamW
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import roc_curve, roc_auc_score, auc,precision_score, f1_score, accuracy_score, recall_score
from datasets import Dataset
import argparse
import ml_metrics as metrics
from bwf import BertWithFeatures
import ast
import random, copy


    
    
def generate_one_episode_from_source(df,
                        
                        support_num_per_class,
                        query_num_per_class,
                        generate_from,
                        promptid=None,
                        avoid_this_prompt_sample_rest=None,                        
                        ):
    '''
    INPUT: one dataframe object

    OUTPUT FORMAT:
    Res = {supportset:dict{0,1,2,3}, queryset:dict{0,1,2,3}}
    Here 0 1 2 3 are all dataframes, each containing one dataframe like the input excel
    promptid = 1
    support_num_per_class = 2
    query_num_per_class = 2
    generate_from = 'train'

    '''
    
    if promptid is not None:
        pass
    elif promptid is None and avoid_this_prompt_sample_rest is not None:
        df = df[df.promptid!=avoid_this_prompt_sample_rest]
        
        # option 1
        promptid = random.choice(list(set(df['promptid']))) 
        
        '''Other Option
        
        
        
        Science = set([1,2,10])
        ELA = set([3,4])
        Biology = set([5,6])
        English = set([7,8,9])
        
        dict1 = {'Science':Science, 'ELA':ELA, 'Biology'=biology, 'English':English}
        dict2 = {   
                    1:'Science',2:'Science',
                    10:'Science',3:'ELA', 4:'ELA', 
                    5:'Biology',6:'Biology',
                    7:'English',8:'English',9:'English'
                }
        subject = dict2[ int(avoid_this_prompt_sample_rest) ]
        s = copy.deepcopy(dict1[subject]) #identical to  s = dict1[subject]
        s.remove( int(avoid_this_prompt_sample_rest) )
        promptid = random.choice(list( s )) 
               
        '''
    
    max_score = max(df[df.promptid==promptid]['score1'])
    min_score = min(df[df.promptid==promptid]['score1'])

    support = dict()
    query = dict()

    for i in range(min_score, max_score + 1):
        if generate_from not in ('valid','train'):
            tmp = df[
                (df.promptid == promptid)
                &
                (df.score1 == i)
                ].sample(support_num_per_class + query_num_per_class) 
        else:      
            tmp = df[
                    (df.promptid == promptid)
                    &
                    (df.train_valid == generate_from) 
                    &
                    (df.score1 == i)
                    ].sample(support_num_per_class + query_num_per_class)

        #ensure support set has no overlap with query set
        tmp1 = tmp[:support_num_per_class]
        tmp2 = tmp[support_num_per_class:]

        support[i] = tmp1
        query[i] = tmp2
  
    return support, query    


"---------------------------------------------------------------------------------------------"

def train_model(model,               
                learning_rate,
                train_dataloader,           
                tokenizer,                           
                num_support,
                num_query,
                use_finetune,
                num_episodes_valid,
                num_episodes,
                best_model,
                num_episodes_test,                     
                device = torch.device("cpu"),                      
                targetprompt = 1 # train with other prompt to predict on this prompt
                ):
    
    optimizer = AdamW(model.parameters(), lr = learning_rate)
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer
        #num_warmup_steps=0,
        #num_training_steps=num_episodes
    )

    model.to(device)    
    counter = 0
    #best_model = best_model
    best_loss_so_far = None
    
    for i in range(num_episodes):
        ta = time.time()
        counter += 1
        model.train()
        
        
        support, query = generate_one_episode_from_source(train_dataloader,
                           promptid = None,
                           avoid_this_prompt_sample_rest = targetprompt,
                           support_num_per_class = num_support,
                           query_num_per_class = num_query,
                           generate_from ='train'
                           )
                           
        logits4oneepisode, prediction, lbl, loss = model.myforward(
                                               support=support, query=query, tokenizer=tokenizer,device=device
                                                )    
      
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
          
        if counter in (500,600,700,800,900,1000):
            print('time cost:', time.time()-ta)
            ta = time.time()
        
            print('---------------Traing Current Episode {}------------------'.format(i+1))
        
            loss, support_ids, query_ids, prediction_list, label_list, pid_list = eval_model(model = model, eval_dataloader = train_dataloader, 
                                    tokenizer = tokenizer, device = device,
                                    avoid_this_prompt_test_others = targetprompt, 
                                    support_num_per_class = num_support, query_num_per_class = num_query,
                                    test_over_n_episodes = num_episodes_valid, use_finetune=0
            )
                      
            if best_loss_so_far is None or best_loss_so_far > loss:
                old_loss = best_loss_so_far
                best_loss_so_far = loss
                if os.path.exists(best_model):
                    os.remove(best_model)
                    print('---old model removed---')
                print('---Best loss has been updated from {} to {}---, new best model saved'.format(round(old_loss,3) if old_loss is not None else None, round(best_loss_so_far,3)))
                torch.save(model,best_model)

            else:
                print('---Old loss {} < new loss {}---'.format(round(best_loss_so_far,3), round(loss,3)))
                print('No improvement detected compared to last validation round, early stop is triggered.')
                model = torch.load(best_model)
                break
                           
    print('-------------Testing on unseen prompt now---------------------')            
    loss, support_ids, query_ids, prediction_list, label_list, pid_list = eval_model(model = model, eval_dataloader = train_dataloader, 
                                    tokenizer = tokenizer, device = device,
                                    #avoid_this_prompt_test_others = targetprompt, 
                                    support_num_per_class = num_support, query_num_per_class = num_query,
                                    test_over_n_episodes = num_episodes_test,
                                    targetprompt = targetprompt, testing = True, use_finetune=use_finetune
                                    #optimizer = optimizer, lr_scheduler = lr_scheduler
            )
        
    df =pd.DataFrame(index=None)
    df['pid']=pd.Series(pid_list)
    df['label']=pd.Series(label_list)
    df['prediction']=pd.Series(prediction_list)
    df['query_ids']=pd.Series(query_ids)
    df['support_ids']=pd.Series(support_ids)
        
    return df
    
    
    
    
    
    
def eval_model(model,                     
                eval_dataloader,          
                tokenizer,
                device = torch.device("cpu"),
                targetprompt = None, # train with other prompt to predict on this prompt
                avoid_this_prompt_test_others = None,
                support_num_per_class = None,
                use_finetune=None,
                query_num_per_class = None,
                test_over_n_episodes = None,
                testing =False#, optimizer=None, lr_scheduler=None
               ):
    t1 = time.time()           
    model.to(device)    
    loss_list = []
    accuracy_list = []
    QWK_list = []
    
    support_ids = []
    query_ids = []
    label_list = []
    prediction_list=[]
    pid_list = []
    
    
    if use_finetune ==1:
        optimizer = AdamW(model.parameters(), lr = 0.0001)
        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer
        ) 
        model_bk = copy.deepcopy(model.state_dict())
    
    for i in range(test_over_n_episodes):       
        
        support, query = generate_one_episode_from_source(eval_dataloader,
                           promptid = targetprompt,
                           avoid_this_prompt_sample_rest = avoid_this_prompt_test_others,
                           support_num_per_class = support_num_per_class,
                           query_num_per_class = query_num_per_class,
                           generate_from = 'valid'if testing is False else 'both'
                           )
                           
        if use_finetune ==1:
            model.train()
            model.myforward_fine_tune_support(
                support = support, 
                tokenizer = tokenizer,device = device,
                optimizer = optimizer, lr_scheduler = lr_scheduler
            )  
            
                           
        model.eval()                   
        with torch.no_grad():                   
            logits4oneepisode, prediction, lbl, loss, episode_info = model.myforward_eval(
                                support = support, query = query, tokenizer = tokenizer,device = device
            )
                                
            loss_list.append(loss)
            accuracy_list.append(round(accuracy_score(lbl,prediction),5))
            QWK = round(metrics.quadratic_weighted_kappa(lbl,prediction),5)
            QWK_list.append(QWK)
            
            support_ids.append(episode_info['support_id'])
            query_ids.append(episode_info['query_id'])
            label_list.append(lbl)
            prediction_list.append(prediction)
            pid_list.append(episode_info['promptid'])
            
        if use_finetune == 1:
            model.load_state_dict(copy.deepcopy(model_bk)) # model para changes must be related to optimizer
            optimizer = AdamW(model.parameters(), lr = 0.0001)# model para changes must be related to optimizer
            lr_scheduler = get_scheduler(                   # model para changes must be related to optimizer
                "constant",
                optimizer=optimizer
            )         
            
    print('-------------Average_Eval_Results---------------------')
    print('Avg Accuracy: ', round(np.mean(accuracy_list), 3) )
    print('Avg QWK: ', round(np.mean(QWK_list), 3))
    print('Avg Loss: ',round(np.mean(loss_list), 3) )
    print('Time cost for evaluation: {} secs'.format(int(time.time()-t1)))
            
    return np.mean(loss_list), support_ids, query_ids, prediction_list, label_list,pid_list #, episode_info
    
  
if __name__ == "__main__":
    time1 = int(time.time())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
    
    parser = argparse.ArgumentParser(description="This is a description")
    parser.add_argument('--num_episodes',dest='num_episodes',required = True,type = int)
    parser.add_argument('--num_episodes_valid',dest='num_episodes_valid',required = True,type = int)
    parser.add_argument('--num_episodes_test',dest='num_episodes_test',required = True,type = int)
    parser.add_argument('--learning_rate',dest='learning_rate',required = True,type = float)    
    parser.add_argument('--train_file',dest='train_file',required = True,type = str)
    parser.add_argument('--best_model',dest='best_model',required = True,type = str)   
    parser.add_argument('--targetprompt', dest='targetprompt', required = True, type = int)
    parser.add_argument('--num_support',dest='num_support',required = True,type = int)
    parser.add_argument('--num_query',dest='num_query',required = True,type = int)
    parser.add_argument('--use_finetune',dest='use_finetune',required = True, type = int)
    parser.add_argument('--result_file',dest='result_file',required = True,type = str)
    parser.add_argument('--load_from_existing_model', dest='load_from_existing_model', required = False, type = str)
    #parser.add_argument('--existing_model',dest='existing_model',required = False,type = str)
    
    args = parser.parse_args()  

    
    num_episodes = args.num_episodes
    num_episodes_valid = args.num_episodes_valid
    num_episodes_test = args.num_episodes_test
    learning_rate = args.learning_rate 
    train_file = args.train_file
    best_model = args.best_model
    targetprompt = args.targetprompt
    num_support = args.num_support
    num_query = args.num_query
    use_finetune = args.use_finetune # 1 means use; 0 means not use
    result_file = args.result_file
    load_from_existing_model = args.load_from_existing_model
    

    print('best_model: {}'.format(best_model))
    print('train_file: {}'.format(train_file)) 
    print('num_episodes: {}'.format(num_episodes))
    print('num_episodes_valid: {}'.format(num_episodes_valid))
    print('num_episodes_test: {}'.format(num_episodes_test))
    print('learning_rate: {}'.format(learning_rate))
    print('targetprompt: {}'.format(targetprompt))
    print('num_support: {}'.format(num_support))
    print('num_query: {}'.format(num_query))
    print('use_finetune: {}'.format('True' if use_finetune==1 else 'False'))
    print('result_file: {}'.format(result_file))
    print('load_from_existing_model: {}'.format(load_from_existing_model))
    
 
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    df = pd.read_excel(train_file)
    
    if load_from_existing_model is None:
        ItNoMatter = 4  # this para has no use and thus be set randomly
        m = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels = ItNoMatter) 
        model = BertWithFeatures(BertCLS = m)
    else:
        model = torch.load(load_from_existing_model)
        

    df = train_model(model = model,
                targetprompt = targetprompt,
                learning_rate = learning_rate, 
                train_dataloader = df,
                device = device,
                num_support = num_support,
                num_query = num_query,
                use_finetune = use_finetune,
                num_episodes_valid = num_episodes_valid,
                num_episodes_test = num_episodes_test,
                num_episodes = num_episodes,
                best_model=best_model,
                tokenizer = tokenizer
                ) 
    
    df.to_excel(result_file, index=None) 
    print('total time: ',int(time.time())-time1,' seconds')



