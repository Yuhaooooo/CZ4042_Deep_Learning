1. Set up environment

	1.1 To install all the dependencies required, please execute the following command in the terminal in the directory of 'cz4042_LiGuanlong_Heyuhao_P2/': 

		'''
		pip install -r requirements.txt
		'''

		'''
		python -c "import nltk; nltk.download('punkt')"
		'''

	1.2 CUDA Toolkit 10.0 and cuDNN v7.6.5 are required to run Python scripts which utilize GPU for training models

	1.3 Run
		'''
		python assignment2_question_b/other/utils/utils.py $ABSOLUTE_PATH_OF_DATA_DIRECTORY
		'''
		Eg. 'python assignment2_question_b/other/utils/utils.py ~/foo/Desktop'
 		To process the csv and store the npy in 'cz4042_LiGuanlong_Heyuhao_P2/assignment2_question_b/other/npy/' for other questions in part b


2. Execution

	2.1 Assignment Part A:

		The Python scripts for part A are in directory 'cz4042_LiGuanlong_Heyuhao_P2/assignment2_question_a/'

		For question 1-3, run the following command (replace the $X by the question number)

		'''
		python assignment2_question_a/Part_A_Question_$X.py
		'''

		The output of Python scripts include npy files and image files:
			The npy files are stored in 'cz4042_LiGuanlong_Heyuhao_P2/assignment2_question_a/others/npy/'
			The image files are stored in 'cz4042_LiGuanlong_Heyuhao_P2/assignment2_question_a/others/Figures'



	2.2 Assignment Part B:


		The Python scripts for part B are in directory 'cz4042_LiGuanlong_Heyuhao_P2/assignment2_question_b/'

		For question 1-6, run the following command

		'''
		python assignment2_question_b/$X.py
		'''

		run 'ls assignment2_question_b/' to see the py files and replace it with $X.py

		The output of Python scripts include npy files and image files:
			The npy files are stored in 'cz4042_LiGuanlong_Heyuhao_P2/assignment2_question_b/other/npy/'
			The image files are stored in in 'cz4042_LiGuanlong_Heyuhao_P2/assignment2_question_b/other/figure'









