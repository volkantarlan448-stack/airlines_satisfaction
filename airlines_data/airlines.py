import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df  = pd.read_csv("train.csv")


print(df.shape)
print(df.head())
print(df.info())
print(df.columns)
print(df.isnull().sum())

plt.figure(figsize=(6,4))
sns.countplot(x="satisfaction",data = df)
plt.title("Hedef Değişken Dağılımı")
plt.show()

print("\nHedef değişken dağılımı")
print(df["satisfaction"].value_counts(normalize=True))
oranlar = df["satisfaction"].value_counts(normalize=True) * 100
oranlar.plot(kind="bar",color =["red","blue"])
plt.title("yolcu memnuniyet oranları (%)")
plt.ylabel("yüzde (%)")
plt.show()

print(df.describe())

df.drop(columns=["Unnamed: 0", "id"],inplace=True)

df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)

df["satisfaction"] = df["satisfaction"].map({
     "neutral or dissatisfied":0,
     "satisfied":1

})

categorical_col = [
        "Gender",
        "Customer Type",
        "Type of Travel",
        "Class"
]

df = pd.get_dummies(df ,columns=categorical_col,drop_first=True)


print(df.shape)
print(df.isnull().sum())
print(df.describe())

df["has_delay"] = ((df["Departure Delay in Minutes"] > 0) | 
                   (df["Arrival Delay in Minutes"] > 0)).astype(int)

df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]

service_cols = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness"

]

df["service_score_avg"] = df[service_cols].mean(axis=1)
print("Servis ortalaması:")
print(df["service_score_avg"])

df["digital_service_avg"] = df[["Inflight wifi service", "Ease of Online booking", "Online boarding"]].mean(axis=1)
print("dijital Servis ortalaması:")
print(df["digital_service_avg"])


df["comfort_service_avg"] = df[["Seat comfort", "Leg room service", "Inflight entertainment", "Food and drink"]].mean(axis=1)
print("confor Servis ortalaması:")
print(df["comfort_service_avg"])


df["operation_service_avg"] = df[["Checkin service", "On-board service", "Inflight service",  "Baggage handling", "Cleanliness"]].mean(axis=1)
print("operasyonel ortalaması:")
print(df["operation_service_avg"])

df["age_group"] = pd.cut(
    df["Age"],
    bins= [0,18,30,50,100],
    labels=["child", "young", "adult", "senior"]
)

df = pd.get_dummies(df,columns=["age_group"],drop_first=True)

print("Yeni veri boyutu:", df.shape)
print(df.head())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

X = df.drop("satisfaction",axis = 1)
y = df["satisfaction"]



X_train  , X_test ,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y )
"""
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train,y_train)

y_pred_lr = log_reg.predict(X_test)
y_proba_lr = log_reg.predict_proba(X_test)[:,1]

print(" logistic regression")
print("accuracy score:", accuracy_score(y_test,y_pred_lr))
print("roc - auc score:",roc_auc_score(y_test,y_proba_lr))
print(classification_report(y_test,y_pred_lr))

plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test,y_pred_lr),
            annot=True,fmt = "d",cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("tahmin")
plt.ylabel("gerçek")
plt.show()

log_reg = LogisticRegression(max_iter=2000)

param_grid_lr = {
    "C":[0.01,0.1,1,10],
    "penalty":["l2"],
    "solver":["lbfgs"]


}

grid_lr = GridSearchCV(
    estimator=log_reg,
    param_grid = param_grid_lr,
    cv = 5,
    scoring="roc_auc",
    n_jobs=1

)

grid_lr.fit(X_train,y_train)

print("Best Logistic Regression Params:")

best_lr = grid_lr.best_estimator_

y_pred_lr = best_lr.predict(X_test)
y_proba_lr = best_lr.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr))
"""
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=1
)

rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:,1]

print("random forest")
print("acuracy:",accuracy_score(y_test,y_pred_rf))
print("roc - auc:", roc_auc_score(y_test,y_proba_rf))
print(classification_report(y_test,y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(4,3))
sns.heatmap(cm,
            annot=True, fmt="d", cmap="Greens",
            xticklabels=["tahmin:mutsuz","tahmin:mutlu"],
            yticklabels=["gerçek:mutsuz","gerçek:mutlu"])
plt.title("Model Tahmin Başarı Matrisi")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show()

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns

).sort_values(ascending=False)

top_10 = importance.head(10)
print(importance.head(10))

plt.figure(figsize=(10,6))
top_10.plot(kind="barh",color= '#2A9D8F')
plt.title("müşteri memnuniyetini belirleyen en kritik 10 faktör")
plt.xlabel("önem dercesi")
plt.ylabel("hizmet")
plt.show()





















for col in df.columns:
    print(col)