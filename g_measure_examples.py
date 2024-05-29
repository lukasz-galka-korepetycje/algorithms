import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, \
    precision_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

from performance_measures.g_measure.G_Measure import G_Measure

#################################################################
####################### Example 1 ###############################
################## Anomaly detection ############################
#################################################################

# load dataset
pendigits = pd.read_csv('datasets/anomaly_detection/pendigits.csv', sep=',')
x_train, x_test, y_train, y_test = train_test_split(pendigits.drop('y', axis=1), pendigits['y'], test_size=0.3,
                                                    random_state=42)

# prepare classifier
classifier = IsolationForest(random_state=44)
classifier.fit(x_train, y_train)

# calculate scores or predictions
y_scores = -classifier.decision_function(x_test)
y_predictions = None

# G Measure presentation
g_measure = G_Measure()
# g_measure.fit(y_true=y_test, y_score=y_scores, y_pred=y_predictions,data_estimation_method='histogram',data_estimation_method_params={'bins':10},data_density_estimation_normalization=False)
# g_measure.fit(y_true=y_test, y_score=y_scores, y_pred=y_predictions,data_estimation_method='kde',data_density_estimation_normalization=True, data_balance={0:1.0, 1:1.0, 2:5.0})
g_measure.fit(y_true=y_test, y_score=y_scores, y_pred=y_predictions, data_estimation_method='kde',
              data_density_estimation_normalization=True,
              aggregation_technique='weighted_average',
              aggregation_technique_params={'weights': {0: 1.0, 1: 1.0}},
              data_balance={0: 1.0, 1: 1.0})
g_measure.print_class_measure()
g_measure.print_overall_measure()
g_measure.plot_class_chart(file_name="pendigits", title=False)

#################################################################
####################### Example 2 ###############################
#################### Classification #############################
#################################################################

# load dataset
iris = load_iris()
x = iris.data
y = iris.target
df = pd.DataFrame(x, columns=iris.feature_names)
df['species'] = y
x_train, x_test, y_train, y_test = train_test_split(df.drop('species', axis=1), df['species'], test_size=0.9,
                                                    random_state=42)

# prepare classifier
classifier = DecisionTreeClassifier(random_state=44)
classifier.fit(x_train, y_train)

# calculate scores or predictions
y_scores = None
y_predictions = classifier.predict(x_test)

# G Measure presentation
g_measure = G_Measure()
# g_measure.fit(y_true=y_test, y_score=y_scores, y_pred=y_predictions,data_estimation_method='histogram',data_estimation_method_params={'bins':10},data_density_estimation_normalization=False)
# g_measure.fit(y_true=y_test, y_score=y_scores, y_pred=y_predictions,data_estimation_method='kde',data_density_estimation_normalization=True, data_balance={0:1.0, 1:1.0, 2:5.0})
g_measure.fit(y_true=y_test, y_score=y_scores, y_pred=y_predictions, data_estimation_method='kde',
              data_density_estimation_normalization=True,
              aggregation_technique='weighted_average',
              aggregation_technique_params={'weights': {0: 1.0, 1: 1.0, 2: 1.0}},
              data_balance={0: 1.0, 1: 1.0, 2: 1.0})
g_measure.print_class_measure()
g_measure.print_overall_measure()
g_measure.plot_class_chart(file_name="iris", title=False)

#################################################################
####################### Example 3 ###############################
###################### Thresholds ###############################
#################################################################

# load dataset
iris = load_iris()
x = iris.data
y = iris.target
df = pd.DataFrame(x, columns=iris.feature_names)
df['species'] = y
x_train, x_test, y_train, y_test = train_test_split(df.drop('species', axis=1), df['species'], test_size=0.9,
                                                    random_state=44)

# prepare classifier
classifier = OneClassSVM()
classifier.fit(x_train, y_train)

# calculate scores or predictions
y_scores = classifier.decision_function(x_test)
y_predictions = None

# G Measure presentation
g_measure = G_Measure()
g_measure.fit(y_true=y_test, y_score=y_scores, y_pred=y_predictions, data_estimation_method='kde',
              data_density_estimation_normalization=True)
g_measure.print_class_measure()
g_measure.print_overall_measure()
g_measure.plot_tresholds_chart(file_name="iris_2", title=False)

#################################################################
####################### Example 4 ###############################
################## Anomaly detection ############################
#################################################################

# load dataset
cover = pd.read_csv('datasets/anomaly_detection/cover.csv', sep=',')
x_train, x_test, y_train, y_test = train_test_split(cover.drop('y', axis=1), cover['y'], test_size=0.3,
                                                    random_state=42)

# prepare classifier
classifier = IsolationForest(random_state=45)
classifier.fit(x_train, y_train)

# calculate scores or predictions
y_scores = -classifier.decision_function(x_test)
y_predictions = None

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
y_pred = (y_scores >= optimal_threshold).astype(int)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# threshold from ROC curve and performance metrics
print(f"Optimal threshold: {optimal_threshold}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"F1 score: {f1}")

# plot ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', marker='o',s=100)
plt.annotate('Optimal threshold',
             (fpr[optimal_idx], tpr[optimal_idx]),
             textcoords="offset points",
             xytext=(70,-50),
             ha='center',
             arrowprops=dict(arrowstyle='->,head_width=0.5,head_length=0.7', connectionstyle='arc3,rad=.5', color='black',lw=2))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("plots/cover_ROC_curve.png")
plt.close()

# PR curve
precision, recall, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

# plot PR curve
plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.fill_between(recall, precision, alpha=0.2, color='blue')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.savefig("plots/cover_PR_curve.png")
plt.close()

# G Measure presentation
g_measure = G_Measure()
g_measure.fit(y_true=y_test, y_score=y_scores, y_pred=y_predictions, data_estimation_method='kde',
              data_density_estimation_normalization=True, data_balance={0: 1.0, 1: 1.0})
g_measure.predict(y_scores)
g_measure.print_class_measure()
g_measure.print_overall_measure()
g_measure.print_thresholds()
g_measure.plot_class_chart(file_name="cover", title=False)
g_measure.plot_tresholds_chart(file_name="cover", title=False)

# performance metrics using thresholds from G Measure
y_pred = g_measure.predict(y_scores)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"F1 score: {f1}")

#################################################################
####################### Example 5 ###############################
#################### Classification #############################
#################################################################

# load dataset
students = pd.read_csv('datasets/classification/predict_students.csv', sep=',')
x_train, x_test, y_train, y_test = train_test_split(students.drop('y', axis=1), students['y'], test_size=0.2,
                                                    random_state=42)

# prepare classifier
classifier = KNeighborsClassifier()
classifier.fit(x_train, y_train)

# calculate scores or predictions
y_scores = None
y_predictions = classifier.predict(x_test)
y_predictions_proba = classifier.predict_proba(x_test)

# plot ROC and PR curve One-vs-Rest
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, len(set(y_test)))]
for i in set(y_test):
    y_true_binary = (y_test == i).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_predictions_proba[:, i])
    roc_auc = auc(fpr, tpr)

    # plot ROC curve
    plt.plot(fpr, tpr, color=colors[i], lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.fill_between(fpr, tpr, alpha=0.5, color=colors[i])
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic Curve One-vs-Rest')
    plt.legend(loc="lower right")
    plt.savefig(f"plots/students_ROC_curve{i}.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true_binary, y_predictions_proba[:, i])
    pr_auc = auc(recall, precision)

    # plot PR curve
    plt.plot(recall, precision, color=colors[i], lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.fill_between(recall, precision, alpha=0.5, color=colors[i])
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="best")
    plt.savefig(f"plots/students_PR_curve{i}.png")  # Save each plot
    plt.close()

# G Measure presentation
g_measure = G_Measure()
g_measure.fit(y_true=y_test, y_score=y_scores, y_pred=y_predictions, data_density_estimation_normalization=True)
g_measure.print_class_measure()
g_measure.print_overall_measure()
g_measure.plot_class_chart(file_name="students", title=False)

# performance metrics report
report = classification_report(y_test, y_predictions, digits=3)
print(report)

#################################################################
####################### Example 6 ###############################
#################### Classification #############################
#################################################################

# load dataset
wine = load_wine()
x = wine.data
y = wine.target
df = pd.DataFrame(x, columns=wine.feature_names)
df['species'] = y
x_train, x_test, y_train, y_test = train_test_split(df.drop('species', axis=1), df['species'], test_size=0.3,
                                                    random_state=42)

# prepare classifier
classifier = OneClassSVM()
classifier.fit(x_train, y_train)

# calculate scores or predictions
y_scores = classifier.decision_function(x_test)
y_predictions = None

# plot ROC and PR curve One-vs-Rest
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, len(set(y_test)))]
for i in set(y_test):
    y_true_binary = (y_test == i).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)

    # plot ROC curve
    plt.plot(fpr, tpr, color=colors[i], lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.fill_between(fpr, tpr, alpha=0.5, color=colors[i])
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic Curve One-vs-Rest')
    plt.legend(loc="lower right")
    plt.savefig(f"plots/wine_ROC_curve{i}.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
    pr_auc = auc(recall, precision)

    # plot PR curve
    plt.plot(recall, precision, color=colors[i], lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.fill_between(recall, precision, alpha=0.5, color=colors[i])
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve One-vs-Rest')
    plt.legend(loc="best")
    plt.savefig(f"plots/wine_PR_curve{i}.png")  # Save each plot
    plt.close()

# G Measure presentation
g_measure = G_Measure()
g_measure.fit(y_true=y_test, y_score=y_scores, y_pred=y_predictions, data_estimation_method='kde',
              data_density_estimation_normalization=True)
g_measure.predict(y_scores)
g_measure.print_class_measure()
g_measure.print_overall_measure()
g_measure.print_thresholds()
g_measure.plot_class_chart(file_name="wine", title=False)