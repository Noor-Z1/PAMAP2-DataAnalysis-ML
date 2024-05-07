from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score


class NaiveBayes():
    def __init__(self,dataFrame,cv=10):
        subject_results = {}
        totalAccuracy=0
        totalRecall=0
        totalPrecision=0
        totalF1Score=0

        for subject_id in dataFrame['subject_id'].unique():
            subject_data = dataFrame[dataFrame['subject_id'] == subject_id]

           
            X = subject_data.drop('activityID', axis=1)
            y = subject_data['activityID']

            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

            fold_metrics = {'accuracy': [], 'precision_macro': [], 'recall_macro': [], 'f1_macro': []}

            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model = GaussianNB()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro')  # Macro averaging
                recall = recall_score(y_test, y_pred, average='macro')    # Macro averaging
                f1 = f1_score(y_test, y_pred, average='macro')           # Macro averaging

                fold_metrics['accuracy'].append(accuracy)
                fold_metrics['precision_macro'].append(precision)
                fold_metrics['recall_macro'].append(recall)
                fold_metrics['f1_macro'].append(f1)

            subject_results[subject_id] = {
                'mean_accuracy':  sum(fold_metrics['accuracy']) / cv,
                'mean_precision_macro': sum(fold_metrics['precision_macro']) / cv,
                'mean_recall_macro':    sum(fold_metrics['recall_macro']) / cv,
                'mean_f1_macro':        sum(fold_metrics['f1_macro']) / cv
            }

            print(f"Subject {subject_id} - Mean Accuracy: {subject_results[subject_id]['mean_accuracy']:.4f}")
            print(f"Subject {subject_id} - 'mean_precision_macro: {subject_results[subject_id]['mean_precision_macro']:.4f}")
            print(f"Subject {subject_id} - 'mean_recall_macro: {subject_results[subject_id]['mean_recall_macro']:.4f}")
            print(f"Subject {subject_id} - 'mean_f1_macro: {subject_results[subject_id]['mean_f1_macro']:.4f}")

            totalAccuracy=totalAccuracy+subject_results[subject_id]['mean_accuracy']
            totalPrecision=totalPrecision+subject_results[subject_id]['mean_precision_macro']
            totalRecall=totalRecall+subject_results[subject_id]['mean_recall_macro']
            totalF1Score=totalF1Score+subject_results[subject_id]['mean_f1_macro']


        print("Using Naive Bayes: ")
        print(f"Average accuracy across all subjects is - {(totalAccuracy/9):.4f}")
        print(f"Average precision across all subjects is - {(totalPrecision/9):.4f}")
        print(f"Average recall across all subjects is - {(totalRecall/9):.4f}")
        print(f"Average F1-score across all subjects is - {(totalF1Score/9):.4f}")

        
        



        