import numpy as np
import sys




predictions = np.array([])


#data_labels = [np.argmax(j) for j in data_labels][int(20400*0.7):]
data_labels = []
num_labels = 51
threshold = 0.65
def GetEER(predict, labels):
	global predictions
	global data_labels
	predictions = predict
	data_labels = [np.argmax(j) for j in labels]
	return CalcEER(0.0, 100, maxDepth=2)

def CalcStats(prediction, actual_user):
	TP, TN, FP, FN = 0, 0, 0, 0
	if(prediction[actual_user] > threshold):
		TP = 1
	else:
		FN = 1
	predicted_users = np.sum(prediction > threshold)
	TN = num_labels - predicted_users - FN
	FP =  predicted_users - TP

	return TP, TN, FP, FN

#print(np.argmax(predictions[0]), data_labels[0])
#print(CalcStats(predictions[0], data_labels[0]))
def CalcEER(thresholdVal, precise, depth=0, maxDepth = 2, EER = -10):
	global threshold
	threshold = thresholdVal
	if depth >= maxDepth:
		print(EER)
		return EER
	EERVal0 = -1000
	EERValMinusOne = -2000
	for k in range(100):
		print("                                   ", end='\r')
		print(threshold, end='\r')
		ans = list(map(CalcStats, predictions, data_labels))
		#print(ans, predictions, data_labels)
		TP, TN, FP, FN = [sum(i) for i in zip(*ans)]
		values = [TP, TN, FP, FN]
		TPR = TP/(TP+FN)
		FNR = FP/(FP+TN)
		EERVal = abs(1-TPR-FNR)
		#print(EERVal, EERVal0, EERValMinusOne, (k-1)/100)
		if EERVal > EERVal0 and EERVal0 < EERValMinusOne:
			#print(EERVal, EERVal0, EERValMinusOne, threshold-(2/precise))
			break
		EERValMinusOne = EERVal0
		EERVal0 = EERVal
		threshold += 1/precise
	return CalcEER(threshold-(2/precise), precise*100, depth=depth+1, maxDepth=maxDepth, EER=FNR)
