# Automatic-Syntex-Generation-using-Transfer-Learning
Natural Language to SysML Code Generation Task


This is my master's research topic about find and implement the state-of-the art method for natural language to SysML code generation task, I've developed theTransformer-based Model with the help of transfering the knowledge of pre-trained Large Language Models (LLMs): BERT and  CodeBERT. I achieved the average test loss of 0.16 and the BLEU Score of 0.85 and CodeBLEU score of 0.97. 


Number of Epochs: 55 (For the exact results mentioned here)
Batch Size: 32 (For the exact results mentioned here)

**Model Architecture**
Below I have attached my Encode-Decoder Model Architecture, which performed well
![Model Architecture](https://github.com/aa-dish/Automatic-Syntex-Generation-using-Transfer-Learning/assets/53014490/be8655b7-4d72-4236-8583-2226620eae8d)


**Execution steps:**
1. Install the all requirements mentioned here
2. Load the BERT_CodeBERT_Code_Generation_Model.py file in your IDE
3. Change the paths of the dataset file and also make the changes in the path of checkpoints, mentioned the path where you want to save your model
4. Run the model


**Future Scope:**
1. Remove the extra space from the .xlsx file where required to remove (My observation after getting the loss of 0.16)
2. Make the proper space between two words (My observation after getting the loss of 0.16)
3. Implement the model with other pre-trained model in order optimize it in more better way
4. One suggested combination can be T5 and CodeT5 models
5. To create an instance, used this trained model as a pre-trained model with the sufficient new data. To create a new data you may use the data augmentation techniques

If you are interested in this topic and would like to read the detail methodology and implementation please contact me, I will share my thesis report.
