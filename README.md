# GeneralizableASAS
Code for AIED2023 Paper: Generalizable Automatic Short Answer Scoring via Prototypical Neural Network

The following is the command within the 'script.sh' file in the "PNN-Bert-All" directory, with the commands, one can see examples of how to run the proposed generalizable ASAS method ("PNN-Bert-All"). 

#---------------------Initial training (when you have no saved models)---------------------

#with this example, you will train a PNN-BERT-ALL （our proposed model） model with 1000 episodes (at most 1000, because we adopt early-stop)

#num_support denotes the number of shots of support instances for each classes

#the trained model will be saved as 1.bert, with which you can directly use for testing different number of shots.

#--targetprompt 1 means now prompt 1 is the target prompt we need to score. (Note that we train with other prompts but test on prompt 1)

#--num_episodes_valid 200 means the number of episodes for validating

#--num_episodes_test 200 means the number of episodes for testing

#please ignore "--use_finetune 0".


python -u code.py --num_episodes 1000 --num_episodes_valid 200 --num_episodes_test 200 --learning_rate 0.00001 --train_file data.xlsx --best_model 1.bert --targetprompt 1 --num_support 5 --num_query 5 --use_finetune 0 --result_file test_1.xlsx > result_1.txt




#-------IF you have already got the saved model----

#and you would like to skip the training 

#process and directly use the saved model for testing, please use the following command：


#Here --num_episodes ==0 means we don't train over the existing saved model.

#--load_from_existing_model 1.bert means the saved model to be used.

#And we specify the number of support shots to be 10 (more shots means better prototypes)

python -u code.py --num_episodes 0 --num_episodes_valid 200 --num_episodes_test 200 --learning_rate 0.00001 --train_file data.xlsx --best_model 1.bert --targetprompt 1 --num_support 10 --num_query 5 --use_finetune 0  --load_from_existing_model 1.bert --result_file test_1.xlsx >shot_10/result_1.txt

