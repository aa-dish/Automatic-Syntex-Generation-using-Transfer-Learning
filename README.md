# Automatic-Syntex-Generation-using-Transfer-Learning
Natural Language to SysML Code Generation Task (My master's research)


This is my master's research topic about find and implement the state-of-the art method for natural language to SysML code generation task, I've developed theTransformer-based Model with the help of transfering the knowledge of pre-trained Large Language Models (LLMs): BERT and  CodeBERT. I achieved the average test loss of 0.16 and the BLEU Score of 0.85 and CodeBLEU score of 0.97. 


**Requirements:**
Transformers (Hugging Face)
Pandas
Sklearn
PyTorch
import openpyxl (To read the excel dataset)

Number of Epochs: 55 (For the exact results mentioned in the report)
Batch Size: 32 (For the exact results mentioned in the report)


**Execution steps:**
1. Install the all requirements mentioned here
2. Load the BERT.py file in your IDE
3. Change the paths of the dataset file and also make the changes in the path of checkpoints, mentioned the path where you want to save your model
4. Run the model


**Future Scope:**
1. Remove the extra space from the .xlsx file where required to remove (My observation after getting the loss of 0.16)
2. Make the proper space between two words (My observation after getting the loss of 0.16)
3. Implement the model with other pre-trained model in order optimize it in more better way
4. One suggested combination can be T5 and CodeT5 models


**Note:**
I've also attached my master's thesis report here, You can use it to get more deeper idea about this model and my research work.
