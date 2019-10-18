1. Directory Structure:

----cz4042_LiGuanlong_Heyuhao_P1

	----assignment1_question_a

		----Part_A_Question_1.py
		----Part_A_Question_2.py
		----Part_A_Question_3.py
		----Part_A_Question_4.py
		----Part_A_Question_5.py

			train.py
			ctg_data_cleaned.csv

			others
				Figures
					Q1.png       
					Q2ab.png      
					Q2ab_Time.png   
					Q2c.png 
					Q3ab.png   
					Q3c.png             
	             	Q4ab.png
	                Q4c.png
	                Q5.png
				Jupyter Notebook Files 
					Part_A_Question_1.ipynb 
					Part_A_Question_2.ipynb
					Part_A_Question_3.ipynb 
					Part_A_Question_4.ipynb
					Part_A_Question_5.ipynb		 
				npy
					train_data.npy
					test_data.npy  

	----assignment1_question_b
		----Part_B_Question_1.py
		----Part_B_Question_2.py
		----Part_B_Question_3.py
		----Part_B_Question_4.py

			others
				admission_predict.csv
				question_b.ipynb
				plot
					q1
						10000epoch.png  
						earlystop.png   
						random50mse.png
					q2
						correlationMap.png
					q3
					q4
						model_comparision.png
				npy
					X_test.npy   
					X_test2.npy  
					X_train.npy  
					X_train2.npy 
					y_test.npy   
					y_train.npy


	----README.txt
	----requirements.txt



2. Process:

	2.1 To install all the dependencies required, please execute the following command in the terminal in the directory 	of 'cz4042_LiGuanlong_Heyuhao_P1/': 

		'''
		pip install -r requirements.txt
		'''

	2.2 Assignment Part A:

		The Python scripts for part A are in directory 'cz4042_LiGuanlong_Heyuhao_P1/assignment1_question_a/'

		For question 1-5, run the following command (replace the X by the question number)

		'''
		python Part_A_Question_X.py
		'''

		The output of Python scripts include npy files and image files:
			The npy files are stored in 'cz4042_LiGuanlong_Heyuhao_P1/assignment1_question_a/others/npy/'
			The Jupyter Notebook Files are stored in 'cz4042_LiGuanlong_Heyuhao_P1/others/Jupyter Notedbook Files/'
			The image files are stored in 'cz4042_LiGuanlong_Heyuhao_P1/assignment1_question_a/others/Figures'



	2.3 Assignment Part B:

		The Python scripts for part B are in directory 'cz4042_LiGuanlong_Heyuhao_P1/assignment1_question_b/'

		For question 1-5, run the following command (replace the X by the question number)

		'''
		python Part_A_Question_X.py
		'''

		The output of Python scripts include npy files and image files:
			The npy files are stored in 'cz4042_LiGuanlong_Heyuhao_P1/assignment1_question_b/others/npy/'
			The Jupyter Notebook Files are stored in 'cz4042_LiGuanlong_Heyuhao_P1/others/question_b.ipynb'
			The image files are stored in 'cz4042_LiGuanlong_Heyuhao_P1/assignment1_question_a/others/plot/'









