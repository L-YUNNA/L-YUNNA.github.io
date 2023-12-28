# %%
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from itertools import product

# %%
import xgboost
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import *
import shap

# %%


# %%
morpho_feature = pd.read_csv('/workspace/src/ganglion/CIPO_sm_info_231126.csv')
data = morpho_feature.drop([10], axis=0).reset_index(drop=True)
data

# %%


# %%
# severity labeling

threeG = [2, 2, 2, 2, 1, 0, 0, 1, 0, 2, 0]
twoG = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
#print(len(threeG), len(twoG))

# %%
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

# %%


# %% [markdown]
# ### 1. Two severity group

# %%
Counter(twoG)

# %%
X_data = data.drop(['FileName'], axis=1)
y_data = np.array(twoG)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=4, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %%
z_scaled_X_train = standard_scaler.fit_transform(X_train)
z_scaled_X_test = standard_scaler.transform(X_test)

mm_scaled_X_train = minmax_scaler.fit_transform(X_train)
mm_scaled_X_test = minmax_scaler.transform(X_test)

r_scaled_X_train = robust_scaler.fit_transform(X_train)
r_scaled_X_test = robust_scaler.transform(X_test)

# %%


# %% [markdown]
# ##### **Feature correlation 확인** 
# - 다중공선성 여부

# %%
z_scaled_data = standard_scaler.fit_transform(X_data)
df = pd.DataFrame(z_scaled_data, columns=['GanglionPerMM', 'GanglionArea', 'GanglioncellNum', 'GanglioncellPerMM', 'GanglioncellArea'])
df.head()

# %%
correlation_matrix = df.corr()

# 히트맵 그리기
plt.figure(figsize=(7,6))
Hmap = sns.heatmap(correlation_matrix, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)

Hmap.set_xticklabels(Hmap.get_xticklabels(), rotation=30, ha='right', fontsize=10)
Hmap.set_yticklabels(Hmap.get_yticklabels(), rotation=30, ha='right', fontsize=10)

plt.title("Correlation Heatmap of Features", fontsize=15)
plt.show()

# %%


# %%


# %% [markdown]
# ### 2. XGBoost

# %% [markdown]
# #### 2-1. Training

# %%
def OneHot(test_y):
    ohe = OneHotEncoder()
    ohe_test_y = ohe.fit_transform(test_y.reshape(-1,1)).toarray()
    return ohe_test_y

def PredAndEval(train_X, train_y, test_X, test_y, classifier):
    classifier.fit(train_X, train_y)
    pred = classifier.predict(test_X)
    prob = classifier.predict_proba(test_X)

    ohe_test_y = OneHot(test_y)
    acc = accuracy_score(test_y, pred)
    auc = roc_auc_score(ohe_test_y, prob)
    return pred, prob, acc, auc

def ML_clf(train_X, train_y, test_X, test_y, params, classifier_type):
    history = {'test_acc':[], 'test_auc':[], 'para':[]}

    # 매개변수의 가능한 조합을 생성
    param_combinations = product(*params.values())

    if classifier_type == 'XGB':
        for est, lr, col, gam, sub, lam in param_combinations:
            clf = XGBClassifier(learning_rate=lr, n_estimators=est, colsample_bytree=col, 
                                gamma=gam, subsample=sub, reg_lambda=lam, random_state=42)
                            
            pred, prob, acc, auc = PredAndEval(train_X, train_y, test_X, test_y, clf)
            para = [est, lr, col, gam, sub, lam]

            [history[key].append(value) for key, value in zip(['test_acc', 'test_auc', 'para'], [acc, auc, para])]

    elif classifier_type == 'RF':
        for est, cri, depth, feat, sam in param_combinations:
            clf = RandomForestClassifier(n_estimators=est, criterion=cri, max_depth=depth,
                                         max_features=feat, max_samples=sam, bootstrap=True, random_state=42)
                            
            pred, prob, acc, auc = PredAndEval(train_X, train_y, test_X, test_y, clf)
            para = [est, cri, depth, feat, sam]

            [history[key].append(value) for key, value in zip(['test_acc', 'test_auc', 'para'], [acc, auc, para])]

    return history

# %%
def MLperScale(scaler_type, params, classifier_type):
    global save_path
    
    if scaler_type == 'z':
        hist = ML_clf(z_scaled_X_train, y_train, z_scaled_X_test, y_test, params, classifier_type)
    elif scaler_type == 'mm':
        hist = ML_clf(mm_scaled_X_train, y_train, mm_scaled_X_test, y_test, params, classifier_type)
    elif scaler_type == 'r':
        hist = ML_clf(r_scaled_X_train, y_train, r_scaled_X_test, y_test, params, classifier_type)
    else:
        print(f"There is no {scaler_type} scaler type. Only put 'z', 'mm', or 'r' type.")

    with open(os.path.join(save_path, f"{classifier_type}_{scaler_type}scale_2group_hist.pkl"), "wb") as f:
        pickle.dump(hist, f)
        
    return hist

# %%
save_path = '/workspace/src/ganglion/stage4_ML_results/2group'

if not os.path.exists(save_path):
    os.mkdir(save_path)

# %%
xgb_params = {
    'n_estimators':[50, 70, 100, 200, 300, 400, 500], 
    'learning_rate':[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005], 
    'colsample_bytree':[0.3, 0.5, 0.7, 1.0],
    'gamma':[0, 0.5, 1, 1.5],
    'subsample':[0.3, 0.5, 0.7, 1.0],
    'reg_lambda':[0, 1, 3, 5]
}

# Train
#z_xgb_hist = MLperScale('z', xgb_params, "XGB")   # 이미 저장돼있음
#mm_xgb_hist = MLperScale('mm', xgb_params, "XGB")
#r_xgb_hist = MLperScale('r', xgb_params, "XGB")

# %%


# %% [markdown]
# #### 2-2. Evaluation

# %%
def GetHist(save_path, fname):
    with open(os.path.join(save_path, fname), "rb") as f:
        loaded_hist = pickle.load(f)
    return loaded_hist

# %%
def BestCase(hist_pkl):
    best_acc = np.max(hist_pkl['test_acc'])
    best_auc = np.max(hist_pkl['test_auc'])
    print(f"Best Acc : {best_acc}\nBest AUC : {best_auc}")

    best_acc_idx = hist_pkl['test_acc'].index(best_acc)
    best_auc_idx = hist_pkl['test_auc'].index(best_auc)

    best_acc_para, best_auc_para = hist_pkl['para'][best_acc_idx], hist_pkl['para'][best_auc_idx]
    print(f"Best Acc param : {best_acc_para}\nBest AUC param : {best_auc_para}")

    return best_acc, best_auc, best_acc_para, best_auc_para

# %%
z_xgb_hist = GetHist(save_path, "XGB_zscale_2group_hist.pkl")
mm_xgb_hist = GetHist(save_path, "XGB_mmscale_2group_hist.pkl")
r_xgb_hist = GetHist(save_path, "XGB_rscale_2group_hist.pkl")

# %%
print("\n**Z-score")
_, _, z_xgb_acc_para, z_xgb_auc_para = BestCase(z_xgb_hist) 

print("\n**Min-Max")
_, _, mm_xgb_acc_para, mm_xgb_auc_para = BestCase(mm_xgb_hist) 

print("\n**Robust")
_, _, r_xgb_acc_para, r_xgb_auc_para = BestCase(r_xgb_hist) 

# %%
# BEST ACC 기준
est, lr, col, gam, sub, lam = z_xgb_acc_para
xgb_clf = XGBClassifier(n_estimators=est, learning_rate=lr, colsample_bytree=col, 
                       gamma=gam, subsample=sub, reg_lambda=lam, random_state=42)

pred, prob, acc, auc = PredAndEval(r_scaled_X_train, y_train, r_scaled_X_test, y_test, xgb_clf)
cm = confusion_matrix(y_test, pred)
print(f"Final test acc & auc : {round(acc, 5)}, {round(auc, 5)}")
print(cm)


# %%
# # BEST AUC 기준
# est, lr, col, gam, sub, lam = best_auc_para
# xgb_clf = XGBClassifier(n_estimators=est, learning_rate=lr, colsample_bytree=col, 
#                        gamma=gam, subsample=sub, reg_lambda=lam, random_state=42)

# pred, prob, acc, auc = PredAndEval(z_scaled_X_train, y_train, z_scaled_X_test, y_test, xgb_clf)
# cm = confusion_matrix(y_test, pred)
# print(f"Final test acc & auc : {round(acc, 5)}, {round(auc, 5)}")
# print(cm)

# %%


# %% [markdown]
# ### 3. Random forest

# %% [markdown]
# #### 3-1. Training

# %%
rf_params = {
    'n_estimators':[30, 50, 70, 100, 200, 300, 400, 500, 600], 
    'criterion':['gini', 'entropy'],  
    'max_depth':[1, 3, 5, 7, 9, 11, 15],
    'max_features':['sqrt', None, 0.5, 0.7],
    'max_samples':[0.3, 0.5, 0.7, 1.0]
}

# Train
# z_rf_hist = MLperScale('z', rf_params, "RF")
# mm_rf_hist = MLperScale('mm', rf_params, "RF")
# r_rf_hist = MLperScale('r', rf_params, "RF")

# %%


# %% [markdown]
# #### 3-2. Evaluation

# %%
z_rf_hist = GetHist(save_path, "RF_zscale_2group_hist.pkl")
mm_rf_hist = GetHist(save_path, "RF_mmscale_2group_hist.pkl")
r_rf_hist = GetHist(save_path, "RF_rscale_2group_hist.pkl")

# %%
print("\n**Z-score")
_, _, z_rf_acc_para, z_rf_auc_para = BestCase(z_rf_hist) 

print("\n**Min-Max")
_, _, mm_rf_acc_para, mm_rf_auc_para = BestCase(mm_rf_hist) 

print("\n**Robust")
_, _, r_rf_acc_para, r_rf_auc_para = BestCase(r_rf_hist) 

# %%
# BEST ACC 기준
est, cri, depth, feat, sam = z_rf_acc_para
rf_clf = RandomForestClassifier(n_estimators=est, criterion=cri, max_depth=depth,
                                max_features=feat, max_samples=sam, bootstrap=True, random_state=42)

pred, prob, acc, auc = PredAndEval(r_scaled_X_train, y_train, r_scaled_X_test, y_test, rf_clf)
cm = confusion_matrix(y_test, pred)
print(f"Final test acc & auc : {round(acc, 5)}, {round(auc, 5)}")
print(cm)


# %%
print(f"True : {y_test}")
print(f"Pred : {pred}")
print(X_test)
# test set은 순서대로 14_01_0451_HE1, 14_01_0008_HE1, 14_01_0627_HE1, 14_01_0990_HE1
# 14_01_0451_HE1 틀림

# %% [markdown]
# ##### SHAP 확인

# %%
rf_train_explainer = shap.TreeExplainer(rf_clf, data=r_scaled_X_train, model_output='probability')
rf_shap_values_train = rf_train_explainer.shap_values(r_scaled_X_train)

# %%
rf_test_explainer = shap.TreeExplainer(rf_clf, data=r_scaled_X_test, model_output='probability')
rf_shap_values_test = rf_test_explainer.shap_values(r_scaled_X_test)

# %%
features = ['GanglionPerMM', 'GanglionArea', 'GanglioncellNum', 'GanglioncellPerMM', 'GanglioncellArea']

# %%
shap.summary_plot(rf_shap_values_train, r_scaled_X_train, features, class_names=['mild', 'severe'],
                  class_inds='original', max_display=10, show=False)

fig, ax = plt.gcf(), plt.gca()

ax.tick_params(labelsize=11)
ax.set_xlabel("mean(|SHAP value|)\n(average impact on model output magnitude)", fontsize=11)

plt.tight_layout()
#plt.show()
plt.savefig('/workspace/src/ganglion/stage4_ML_results/2group/rf_train_SHAP.png')

# %%
shap.summary_plot(rf_shap_values_test, r_scaled_X_test, features, class_names=['mild', 'severe'],
                  class_inds='original', max_display=10, show=False)

fig, ax = plt.gcf(), plt.gca()

ax.tick_params(labelsize=11)
ax.set_xlabel("mean(|SHAP value|)\n(average impact on model output magnitude)", fontsize=11)

plt.tight_layout()
#plt.show()
plt.savefig('/workspace/src/ganglion/stage4_ML_results/2group/rf_test_SHAP.png')

# %%
for i, cls_name in enumerate(['mild', 'severe']):
    # train
    plt.figure()
    shap.summary_plot(rf_shap_values_train[i], r_scaled_X_train, features, max_display=10, plot_size=(10,6), show=False)
    plt.tight_layout()
    #plt.show()
    plt.savefig('/workspace/src/ganglion/stage4_ML_results/2group/rf_train_' + str(cls_name) + '_beeswarm.png')

# %%
for i, cls_name in enumerate(['mild', 'severe']):
    # test
    plt.figure()
    shap.summary_plot(rf_shap_values_test[i], r_scaled_X_test, features, max_display=10, plot_size=(10,6), show=False)
    plt.tight_layout()
    #plt.show()
    plt.savefig('/workspace/src/ganglion/stage4_ML_results/2group/rf_test_' + str(cls_name) + '_beeswarm.png')

# %% [markdown]
# #### permutation importance

# %%
r = permutation_importance(rf_clf, r_scaled_X_train, y_train, n_repeats=10, random_state=42)

sorted_idx = r.importances_mean.argsort()

for index in sorted_idx:
    print(f"{X_train.columns[index]}: {r.importances_mean[index]}")

plt.barh(range(len(r.importances_mean)), r.importances_mean[sorted_idx])
plt.yticks(range(len(r.importances_mean)), X_train.columns[sorted_idx])  # y축 눈금 설정
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()

#############################
# permutation importance에서 음수값은 해당 feature가 permute 되었을 때, 오히려 성능이 올라갔다는 것 
# (해당 feature을 permute 했을 때, 성능의 증감을 통해서 중요도를 나열하고, 따라서 성능이 많이 떨어진 경우, 해당 feature가 모델 학습에 중요하다는 것)
# 즉, 음수 값을 가지는 변수들은 모델에 영향을 끼치지 못하는 중요하지 않은 feature
# 워낙 작은 dattset이라 더 그러함.. 
#############################

# %%
# test
r = permutation_importance(rf_clf, r_scaled_X_test, y_test, n_repeats=10, random_state=42)

sorted_idx = r.importances_mean.argsort()

for index in sorted_idx:
    print(f"{X_train.columns[index]}: {r.importances_mean[index]}")

plt.barh(range(len(r.importances_mean)), r.importances_mean[sorted_idx])
plt.yticks(range(len(r.importances_mean)), X_train.columns[sorted_idx])  # y축 눈금 설정
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()

#############################
# permutation importance에서 음수값은 해당 feature가 permute 되었을 때, 오히려 성능이 올라갔다는 것 
# (해당 feature을 permute 했을 때, 성능의 증감을 통해서 중요도를 나열하고, 따라서 성능이 많이 떨어진 경우, 해당 feature가 모델 학습에 중요하다는 것)
# 즉, 음수 값을 가지는 변수들은 모델에 영향을 끼치지 못하는 중요하지 않은 feature
# 워낙 작은 dattset이라 더 그러함.. 
#############################

# %% [markdown]
# #### feature importance 
# 
# train set에 대한 것이라고 함

# %%
# 특성 중요도 확인
feature_importance = rf_clf.feature_importances_

# 중요도를 기준으로 내림차순으로 특성의 인덱스 정렬
sorted_idx = np.argsort(feature_importance)

# 각 특성의 중요도와 이름 출력
for index in sorted_idx:
    print(f"{X_train.columns[index]}: {feature_importance[index]}")

# 중요도 시각화
plt.barh(range(len(feature_importance)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_importance)), X_train.columns[sorted_idx])  # y축 눈금 설정
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()

# %%
# BEST AUC 기준
est, cri, depth, feat, sam = r_rf_auc_para
rf_clf = RandomForestClassifier(n_estimators=est, criterion=cri, max_depth=depth,
                                max_features=feat, max_samples=sam, bootstrap=True, random_state=42)

pred, prob, acc, auc = PredAndEval(r_scaled_X_train, y_train, r_scaled_X_test, y_test, rf_clf)
cm = confusion_matrix(y_test, pred)
print(f"Final test acc & auc : {round(acc, 5)}, {round(auc, 5)}")
print(cm)


# %%
# 특성 중요도 확인
feature_importance = rf_clf.feature_importances_

# 중요도를 기준으로 내림차순으로 특성의 인덱스 정렬
sorted_idx = np.argsort(feature_importance)

# 각 특성의 중요도와 이름 출력
for index in sorted_idx:
    print(f"{X_train.columns[index]}: {feature_importance[index]}")

# 중요도 시각화
plt.barh(range(len(feature_importance)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_importance)), X_train.columns[sorted_idx])  # y축 눈금 설정
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()

# %%


# %%


# %%


# %%



