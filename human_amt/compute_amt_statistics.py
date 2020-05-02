import pandas as pd
import os
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict, Counter
import numpy as np


def computer(x):
    if x == 3 or x == 0:
        return 1.0
    elif x == 2 or x == 1:
        return 0.67
    else:
        raise Exception
    
def vote(x):
    if x == 2 or x == 3:
        return 1
    elif x == 0 or x == 1:
        return 0
    else:
        raise Exception

def mapper(x):
    if x == 'Very Difficult':
        return 1.0
    elif x == 'Difficult':
        return 2.0
    elif x == 'Neutral':
        return 3.0
    elif x == 'Easy':
        return 4.0
    elif x == 'Very Easy':
        return 5.0
    
def mapper_conf(x):
    if x == 'Not confident':
        return 1.0
    elif x == 'A little confident':
        return 2.0
    elif x == 'Pretty confident':
        return 3.0
    elif x == 'Very confident':
        return 4.0

if __name__ == "__main__":

	models = []
	scores = []

	for file in os.listdir('./Results/'):
		df = pd.read_csv(open(os.path.join('./Results/', file),'r'))
		name = df['Input.method'].unique()[0]
		
		if name == 'human' or 'random' in name:
			if 'Input.question' in df.columns:
				name = 'multirc_' + name
			else:
				name = 'movies_' + name
		print(name)
		print(file)

		#re-format multirc cols to match (also for human + random)
		if 'Input.question' in df.columns:
			df['Answer.pos.Positive'] = df['Answer.pos.Yes']
			df['Answer.neg.Negative'] = df['Answer.neg.No']
		
		df['prediction'] = [0 if row['Answer.neg.Negative'] else 1 for _, row in df.iterrows()] 
		
		# TEST READABILITY & CONFIDENCE:
		# convert all to binary
		df['Very Difficult'] = [0 if not row['Answer.choiceA.A'] else 1 for _, row in df.iterrows()]
		df['Difficult'] = [0 if not row['Answer.choiceB.B'] else 1 for _, row in df.iterrows()]
		df['Neutral'] = [0 if not row['Answer.choiceC.C'] else 1 for _, row in df.iterrows()]
		df['Easy'] = [0 if not row['Answer.choiceD.D'] else 1 for _, row in df.iterrows()]
		df['Very Easy'] = [0 if not row['Answer.choiceE.E'] else 1 for _, row in df.iterrows()]
		df['Not confident'] = [0 if not row['Answer.opt0.0'] else 1 for _, row in df.iterrows()]
		df['A little confident'] = [0 if not row['Answer.opt1.1'] else 1 for _, row in df.iterrows()]
		df['Pretty confident'] = [0 if not row['Answer.opt2.2'] else 1 for _, row in df.iterrows()]
		df['Very confident'] = [0 if not row['Answer.opt3.3'] else 1 for _, row in df.iterrows()]
		
		# confirm no overlaps
		df['total_read'] = df['Very Difficult'] + df['Difficult'] + df['Neutral'] + df['Easy'] + df['Very Easy']
		df['total_conf'] = df['Not confident'] + df['A little confident'] + df['Pretty confident'] + df['Very confident']
		df.drop(list(df.loc[df['total_read'] != 1].index), inplace=True)
		assert df.loc[df['total_read'] != 1].shape[0] == 0
		assert df.loc[df['total_conf'] != 1].shape[0] == 0
		
		# now do the pivot:
		x = df[['Very Difficult', 'Difficult', 'Neutral', 'Easy', 'Very Easy']].stack()
		df['readability'] = pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1)))
		df['readability_score'] = df['readability'].apply(mapper)
		try :
			df['readability_score'] = df['readability_score'].astype(np.int8)
		except :
			breakpoint()
		
		# do the same for confidence
		x = df[['Not confident', 'A little confident', 'Pretty confident', 'Very confident']].stack()
		df['confidence'] = pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1)))
		df['confidence_score'] = df['confidence'].apply(mapper_conf)
		df['confidence_score'] = df['confidence_score'].astype(np.int8)
		
		#now calculate inter-annotator agreement:
		metrics_by_hit = df.groupby(['HITId', 'Input.label'], as_index=False).agg({"prediction":'sum', "readability_score":'mean', 'confidence_score':'mean'})
		assert len(metrics_by_hit['prediction'].unique()) == 4
		
		# assert each HITId has 3 reviewers
		cnts = df.groupby(['HITId', 'Input.label'], as_index=False).size().reset_index(name='counts')
	#     assert cnts.loc[cnts['counts'] != 3].shape[0] == 0
		
		metrics_by_hit['ie_agreement'] = metrics_by_hit['prediction'].apply(computer)
		print("IE: %s" % metrics_by_hit.describe().loc['mean']['ie_agreement'])
		
		# compute F1 and accuracy of human predictions against gold-labels:
		metrics_by_hit['majority_vote'] = metrics_by_hit['prediction'].apply(vote)
		y_true = list(metrics_by_hit['Input.label'])
		y_pred = list(metrics_by_hit['majority_vote'])
		print(accuracy_score(y_true, y_pred))
		print("F1 Score: %s" % f1_score(y_true, y_pred))
		print("Accuracy Score: %s" % accuracy_score(y_true, y_pred))
		print("Readability: " + "%s +/-%s" % (round(metrics_by_hit['readability_score'].mean(), 2), round(metrics_by_hit['readability_score'].describe()['std'], 2)))
		print("Confidence: " + "%s +/-%s" % (round(metrics_by_hit['confidence_score'].mean(), 2), round(metrics_by_hit['confidence_score'].describe()['std'], 2)))
		print()

		print(f'{accuracy_score(y_true, y_pred)} & {round(metrics_by_hit["confidence_score"].mean(), 2)} \\textpm{round(metrics_by_hit["confidence_score"].describe()["std"], 2)} & {round(metrics_by_hit["readability_score"].mean(), 2)} \\textpm {round(metrics_by_hit["readability_score"].describe()["std"], 2)}')
		
		for el in list(metrics_by_hit['readability_score']):
			models.append(name)
			scores.append(el)