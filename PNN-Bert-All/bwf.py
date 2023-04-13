from datasets import Dataset
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
from torch.utils.data import DataLoader
import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import random
from transformers.modeling_outputs import SequenceClassifierOutput
import gc
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc,precision_score, f1_score, accuracy_score, recall_score

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]



'''

from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)


from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from transformers.utils import logging

from transformers.models.bert.configuration_bert import BertConfig

'''

def generate_n_sample_loader(df,shuffle=True,batch_size = 1,tokenizer=None):
    dataset = Dataset.from_pandas(df)

    tokenized_datasets = dataset.map(
                                lambda x: tokenizer(x["studentanswer"],x["prompt"], padding="max_length", truncation=True)
                                 , batched = True
                                )
                                
    '''
    tokenized_datasets = dataset.map(
                                lambda x: tokenizer(x["studentanswer"],x["prompt"], padding="max_length", truncation=True)
                                 , batched = True
                                )
    '''
    tokenized_datasets = tokenized_datasets.map(
                                    lambda x : {'labels' : x['score1']}
                                    )

    # keep the following cols only
    cols_to_remove = [col for col in tokenized_datasets.features.keys() if col not in ('attention_mask','input_ids','token_type_ids','labels')]
    tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove)   # the above columns are to be matched with forward function in bwf

    tokenized_datasets.set_format("torch")

    dataloader = DataLoader(tokenized_datasets, shuffle=shuffle, batch_size = batch_size) 
    
    return dataloader


class BertWithFeatures(nn.Module):
    def __init__(self, BertCLS=None):
        super().__init__()
        
        self.BertCLS = BertCLS        
        
        #in_features = BertCLS.classifier.in_features
        #out_features = BertCLS.classifier.out_features       
        #self.classifier = nn.Linear(in_features + feature_num, out_features)           
        #self.classifier.weight.data.normal_(mean = 0.0, std = self.BertCLS.config.initializer_range)
        #if self.classifier.bias is not None:
            #self.classifier.bias.data.zero_()
        
        # Here you should initialize the weights of self.classifier  or USE the default initialization.
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
        
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.BertCLS.config.use_return_dict

        outputs = self.BertCLS.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # return the encoded representation of the input text (CLS from bert-encoder)
        return outputs[1].reshape(1,-1)
        
        
        
        
        
    def myforward(self, support, query, tokenizer, device): # from one episode   
        
        logits4oneepisode = []
        loss4episode = []   
        pair_num = len(query.keys())*1.0* len(query[0])* (len(support.keys()) - 1) #query_num * (prototype_num-1)

        for qscore, qdf in query.items():
            q_loader = generate_n_sample_loader(qdf,tokenizer=tokenizer)
            for e in q_loader:
                e = {k: v.to(device) for k, v in e.items()}
                logits = []  #lgots for current query object
                non_tgt = []
                for pscore, pdf in support.items():
                    if int(pscore) ==int(qscore):
                        tgt = (pscore,pdf)
                    else:
                        non_tgt.append((pscore,pdf))
                        
                support_pair = []
                for pscore, pdf in non_tgt:
                    support_pair.append( (tgt[0], tgt[1], pscore, pdf) )
                            
                for target_pscore, target_pdf, pscore, pdf in support_pair: # support_cls_num - 1
                    target_p_loader = generate_n_sample_loader(target_pdf, tokenizer=tokenizer)
                    target_prototype = []
                    for ee in target_p_loader:
                        ee = {k: v.to(device) for k, v in ee.items()}                     
                        tmp = self.forward(**ee)  # tmp = self.forward(**ee)
                        target_prototype.append(tmp)       
                    target_prototype = torch.mean(torch.cat(target_prototype, dim = 0), 0, True) #  ---> representation for this prototype of pscore
                    
                    p_loader = generate_n_sample_loader(pdf, tokenizer=tokenizer)
                    non_target_prototype = []
                    for ee in p_loader:
                        ee = {k: v.to(device) for k, v in ee.items()}             
                        tmp = self.forward(**ee)  # tmp = self.forward(**ee)
                        non_target_prototype.append(tmp)
                    non_target_prototype = torch.mean(torch.cat(non_target_prototype, dim = 0), 0, True) #  ---> representation for this prototype of pscore
                    
                    encoded_q = self.forward(**e)
                    tgt_dist = nn.PairwiseDistance(p=2)(encoded_q,target_prototype) # calculate the distance and then the loss 
                    non_tgt_dist = nn.PairwiseDistance(p=2)(encoded_q,non_target_prototype)
                    pair_loss = (-1)*torch.log( torch.sigmoid(non_tgt_dist-tgt_dist) ) *1.0/ (pair_num)   
                    loss4episode.append(pair_loss.item() * pair_num)
                    pair_loss.backward()
                    
                    if len(logits)==0:
                        logits.append((target_pscore,tgt_dist))
                    logits.append((pscore,non_tgt_dist))

                logits = sorted(logits,key = lambda x:x[0],reverse = False)  # q from c1, c2, c3, c4 distance *-1
                logits = [v*(-1) for k,v in logits]
                logits = torch.stack(logits).view(1,len(support))
                logits4oneepisode.append((qscore,logits))

        lbl = [a for a, b in logits4oneepisode]
        loss4episode = np.mean(loss4episode)
        logits4oneepisode = torch.stack([b for a, b in logits4oneepisode]).view(-1,len(support))
        prediction = torch.argmax(logits4oneepisode, dim=1).cpu().tolist()
        
        return logits4oneepisode, prediction, lbl, loss4episode
        
        
        
        
    def myforward_eval(self, support, query, tokenizer, device): # from one episode   
        
        logits4oneepisode = []
        loss4episode = []   
        pair_num = len(query.keys())*1.0* len(query[0])* (len(support.keys()) - 1) #query_num * (prototype_num-1)
        
        for qscore, qdf in query.items():
            q_loader = generate_n_sample_loader(qdf,tokenizer=tokenizer)
            for e in q_loader:
                e = {k: v.to(device) for k, v in e.items()}
                encoded_q = self.forward(**e)
                logits = []  #lgots for current query object
                non_tgt = []
                for pscore, pdf in support.items():
                    if int(pscore) ==int(qscore):
                        tgt = (pscore,pdf)
                    else:
                        non_tgt.append((pscore,pdf))                            
                    
                target_p_loader = generate_n_sample_loader(tgt[1], tokenizer=tokenizer)
                target_prototype = []
                for ee in target_p_loader:
                    ee = {k: v.to(device) for k, v in ee.items()}                     
                    tmp = self.forward(**ee)  # tmp = self.forward(**ee)
                    target_prototype.append(tmp)       
                target_prototype = torch.mean(torch.cat(target_prototype, dim = 0), 0, True) #  ---> representation for this prototype of pscore
                                                                                                                         
                for pscore,pdf in non_tgt: # support_cls_num - 1                                   
                    p_loader = generate_n_sample_loader(pdf, tokenizer=tokenizer)
                    non_target_prototype = []
                    for ee in p_loader:
                        ee = {k: v.to(device) for k, v in ee.items()}             
                        tmp = self.forward(**ee)  # tmp = self.forward(**ee)
                        non_target_prototype.append(tmp)
                    non_target_prototype = torch.mean(torch.cat(non_target_prototype, dim = 0), 0, True) #  ---> representation for this prototype of pscore
                    
                    tgt_dist = nn.PairwiseDistance(p=2)(encoded_q,target_prototype) # calculate the distance and then the loss 
                    non_tgt_dist = nn.PairwiseDistance(p=2)(encoded_q,non_target_prototype)
                    pair_loss = (-1) * torch.log( torch.sigmoid(non_tgt_dist - tgt_dist) ) * 1.0/ (pair_num)   
                    loss4episode.append(pair_loss.item() * pair_num)
                    #pair_loss.backward()
                    
                    if len(logits)==0:
                        logits.append((tgt[0],tgt_dist))
                    logits.append((pscore,non_tgt_dist))

                logits = sorted(logits,key = lambda x:x[0],reverse = False)  # q from c1, c2, c3, c4 distance *-1
                logits = [v*(-1) for k,v in logits]
                logits = torch.stack(logits).view(1,len(support))
                logits4oneepisode.append((qscore,logits))

        lbl = [a for a, b in logits4oneepisode]
        loss4episode = np.mean(loss4episode)
        logits4oneepisode = torch.stack([b for a, b in logits4oneepisode]).view(-1,len(support))
        prediction = torch.argmax(logits4oneepisode, dim=1).cpu().tolist()
        
        
        episode_info = dict()
        episode_info['promptid'] = list(list(query.values())[0]['promptid'])[0]     
        res = []
        for l in [  list(e['id']) for e in support.values()  ]:
            res+= l
        episode_info['support_id'] = res    
        res = []
        for l in [  list(e['id']) for e in query.values()  ]:
            res+= l
        episode_info['query_id'] = res
        
        
        return logits4oneepisode, prediction, lbl, loss4episode, episode_info
        
        
    

    def self_training_pair_from_support(self, support):
        all_ids = []
        for score,pdf in support.items():
            all_ids+= list(pdf['id'])

        pseudo=[]
        for id in all_ids:
            pseudo_support=dict()
            pseudo_query=dict()
            for score, pdf in support.items():
                if id not in list(pdf['id']):
                    pseudo_support[score]=pdf
                else:
                    pseudo_support[score]=pdf[pdf.id!=id]
                    pseudo_query[score] = pdf[pdf.id==id]
            pseudo.append((pseudo_support,pseudo_query))
        return pseudo
        
    def myforward_fine_tune_support(self, support, tokenizer, device, optimizer, lr_scheduler): # from one episode   
        
        #logits4oneepisode = []
        #loss4episode = []   
        pair_num = len(support.keys())*1.0* len(support[0])* (len(support.keys()) - 1) #query_num * (prototype_num-1)
        
        
        
        #"do something to generate pseudo support and seudo query"
        pseudo = self.self_training_pair_from_support(support)
        
        for psupport, query in pseudo:
        

            for qscore, qdf in query.items():
                q_loader = generate_n_sample_loader(qdf,tokenizer=tokenizer)
                for e in q_loader:
                    e = {k: v.to(device) for k, v in e.items()}
                    #logits = []  #lgots for current query object
                    non_tgt = []
                    for pscore, pdf in psupport.items():
                        if int(pscore) ==int(qscore):
                            tgt = (pscore,pdf)
                        else:
                            non_tgt.append((pscore,pdf))
                            
                    support_pair = []
                    for pscore, pdf in non_tgt:
                        support_pair.append( (tgt[0], tgt[1], pscore, pdf) )
                                
                    for target_pscore, target_pdf, pscore, pdf in support_pair: # support_cls_num - 1
                        target_p_loader = generate_n_sample_loader(target_pdf, tokenizer=tokenizer)
                        target_prototype = []
                        for ee in target_p_loader:
                            ee = {k: v.to(device) for k, v in ee.items()}                     
                            tmp = self.forward(**ee)  # tmp = self.forward(**ee)
                            target_prototype.append(tmp)       
                        target_prototype = torch.mean(torch.cat(target_prototype, dim = 0), 0, True) #  ---> representation for this prototype of pscore
                        
                        p_loader = generate_n_sample_loader(pdf, tokenizer=tokenizer)
                        non_target_prototype = []
                        for ee in p_loader:
                            ee = {k: v.to(device) for k, v in ee.items()}             
                            tmp = self.forward(**ee)  # tmp = self.forward(**ee)
                            non_target_prototype.append(tmp)
                        non_target_prototype = torch.mean(torch.cat(non_target_prototype, dim = 0), 0, True) #  ---> representation for this prototype of pscore
                        
                        encoded_q = self.forward(**e)
                        tgt_dist = nn.PairwiseDistance(p=2)(encoded_q,target_prototype) # calculate the distance and then the loss 
                        non_tgt_dist = nn.PairwiseDistance(p=2)(encoded_q,non_target_prototype)
                        pair_loss = (-1)*torch.log( torch.sigmoid(non_tgt_dist-tgt_dist) ) *1.0/ (pair_num)   
                        #loss4episode.append(pair_loss.item() * pair_num)
                        pair_loss.backward()
                    
                    
                    """
                    if len(logits)==0:
                        logits.append((target_pscore,tgt_dist))
                    logits.append((pscore,non_tgt_dist))

                logits = sorted(logits,key = lambda x:x[0],reverse = False)  # q from c1, c2, c3, c4 distance *-1
                logits = [v*(-1) for k,v in logits]
                logits = torch.stack(logits).view(1,len(support))
                logits4oneepisode.append((qscore,logits))
                    """
                    
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        

        #lbl = [a for a, b in logits4oneepisode]
        #loss4episode = np.mean(loss4episode)
        #logits4oneepisode = torch.stack([b for a, b in logits4oneepisode]).view(-1,len(support))
        #prediction = torch.argmax(logits4oneepisode, dim=1).cpu().tolist()
        
        return None#logits4oneepisode, prediction, lbl, loss4episode        
        
        
        

    def myforward_active(self, support, query, tokenizer, device): # from one episode   
        '''
            for a well_trained model, to test if it can generalize to unseen prompts for short answer scoring
            starting from 1-shot 15 query, gradually label the queries and at the end, we will reach 16-shot, 0query
            in the process, we test the accuracy after each time a human labels an instance
            
            OUTPUT be like: {'Method Accuracy':[a1,a2...an],'Model Accuracy':[a1,a2...an],'Unseen Accuracy':[a1,a2...an-1]}
            *** need to run at least 200 runs (on 200 sets of support set and query set combination) to see what the curve looks like
        '''
        new_support=dict()
        with torch.no_grad(): 
            for pscore, pdf in support.items():                            

                target_p_loader = generate_n_sample_loader(pdf, tokenizer=tokenizer)
                cnt=-1
                for ee in target_p_loader:
                    cnt+=1
                    ee = {k: v.to(device) for k, v in ee.items()}                     
                    tmp = self.forward(**ee)  # tmp = self.forward(**ee)
                    new_support[str(pscore)+'_'+str(cnt)] = tmp

        new_query=dict()
        with torch.no_grad():
            for pscore, pdf in query.items():                            

                target_p_loader = generate_n_sample_loader(pdf, tokenizer=tokenizer)
                cnt=0
                for ee in target_p_loader:
                    cnt+=1
                    ee = {k: v.to(device) for k, v in ee.items()}                     
                    tmp = self.forward(**ee)  # tmp = self.forward(**ee)
                    new_query[str(pscore)+'_'+str(cnt)] = tmp
                    
                                  
        current_support = set(new_support.keys())

        all_result = []

        for i in range(len(new_query)+1):
            current_result = []
            for qnumber,q in new_query.items():

                support_prototype_dict = {}

                for snumber,s in new_support.items():
                    if snumber.split('_')[0] not in support_prototype_dict:
                        support_prototype_dict[snumber.split('_')[0]] = [s]
                    else:
                        support_prototype_dict[snumber.split('_')[0]].append(s)

                logit4q=[] # 单个 q 与  若干个 prototype的欧式距离

                for score, lst in support_prototype_dict.items():

                    prototype = torch.mean(torch.cat(lst, dim = 0), 0, True)
                    dist = (-1)* nn.PairwiseDistance(p = 2)(q,prototype)
                    logit4q.append((score,dist))

                logit4q = sorted(logit4q, key = lambda x:x[0],reverse = False)  # q from c1, c2, c3, c4 distance *-1

                logit4q = [v for k,v in logit4q]
                logit4q = torch.stack(logit4q).view(1,len(logit4q))
                logit4q = torch.softmax(logit4q, dim = 1)

                entropy_q = round(torch.sum(-logit4q.mul(torch.log(logit4q))).item(),5)  # acquisition function, in this case would be entropy_q

                prediction = torch.argmax(logit4q, dim = 1).item()
                current_result.append((
                                      int(qnumber.split('_')[0]), #label
                                      [round(e ,5) for e in list(logit4q.view(-1).numpy())], #logits details
                                      prediction,
                                      1 if qnumber in current_support else 0, #whether this query in support set or not
                                      qnumber,                                # score_number: a identifier of this query instance
                                      entropy_q                               # prediction entropy
                                      ))

            all_result.append(current_result)

            if i< len(new_query):
                selected_query = sorted(
                                        [e for e in current_result if e[-3]==0], 
                                        key=lambda x:x[5], 
                                        reverse=True
                                       )[0]
                # choose maximum uncertainty instance according to acquisition function metric.
                current_support.add(selected_query[-2])
                new_support[selected_query[-2]] = new_query[selected_query[-2]]                
                    
            
        metrics={'Method Accuracy':[],'Model Accuracy':[],'Unseen Accuracy':[]}
        for result in all_result:
            Method_Accuracy = (len( [ d for a,b,c,d,e,f in result if d==1]  )*1.0 + len( [ d for a,b,c,d,e,f in result if (a==c and d==0)]  ))/(len(result))
            Method_Accuracy = round(Method_Accuracy,3)
            metrics['Method Accuracy'].append(Method_Accuracy)

            Model_Accuracy = (len( [ d for a,b,c,d,e,f in result if (a==c and d==1)]  )*1.0 + len( [ d for a,b,c,d,e,f in result if (a==c and d==0)]  ))/(len(result))
            Model_Accuracy = round(Model_Accuracy,3)
            metrics['Model Accuracy'].append(Model_Accuracy)

            if len( [ d for a,b,c,d,e,f in result if (d==0)]  )>0:
                unseen_Accuracy = len( [ d for a,b,c,d,e,f in result if (a==c and d==0)]  )*1.0 /len( [ d for a,b,c,d,e,f in result if (d==0)]  )
                unseen_Accuracy = round(unseen_Accuracy,3)
                metrics['Unseen Accuracy'].append(unseen_Accuracy)
                
        return metrics   # for 15-query, we can report acc every 1 query, then the effect of different scoring range can be addressed
        #you can also generate other results or metrics from 'all_result'


        
