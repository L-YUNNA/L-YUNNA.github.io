{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pickle\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from collections import Counter\n",
    "from itertools import product"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import xgboost\n",
    "from xgboost.sklearn import XGBClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder\n",
    "from sklearn.model_selection import StratifiedKFold, train_test_split\n",
    "from sklearn.inspection import permutation_importance\n",
    "from sklearn.metrics import *\n",
    "import shap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>FileName</th>\n",
       "      <th>GanglionPerMM</th>\n",
       "      <th>GanglionArea</th>\n",
       "      <th>GanglioncellNum</th>\n",
       "      <th>GanglioncellPerMM</th>\n",
       "      <th>GanglioncellArea</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>14_01_0008_HE1</td>\n",
       "      <td>0.006488</td>\n",
       "      <td>1287.321894</td>\n",
       "      <td>1.405797</td>\n",
       "      <td>0.008269</td>\n",
       "      <td>288.188242</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>14_01_0293_HE1</td>\n",
       "      <td>0.008789</td>\n",
       "      <td>1460.626287</td>\n",
       "      <td>1.944954</td>\n",
       "      <td>0.017752</td>\n",
       "      <td>186.015293</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>14_01_0362_HE1</td>\n",
       "      <td>0.004426</td>\n",
       "      <td>465.815917</td>\n",
       "      <td>0.450000</td>\n",
       "      <td>0.002088</td>\n",
       "      <td>182.457583</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>14_01_0396_HE1</td>\n",
       "      <td>0.009005</td>\n",
       "      <td>1206.458839</td>\n",
       "      <td>1.045161</td>\n",
       "      <td>0.007722</td>\n",
       "      <td>326.828353</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>14_01_0413_HE1</td>\n",
       "      <td>0.012737</td>\n",
       "      <td>1228.463400</td>\n",
       "      <td>1.164179</td>\n",
       "      <td>0.013570</td>\n",
       "      <td>146.078345</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>14_01_0451_HE1</td>\n",
       "      <td>0.006948</td>\n",
       "      <td>1721.300619</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.011653</td>\n",
       "      <td>208.471246</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>14_01_0470_HE1</td>\n",
       "      <td>0.002688</td>\n",
       "      <td>1129.185248</td>\n",
       "      <td>2.272727</td>\n",
       "      <td>0.006868</td>\n",
       "      <td>180.986408</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>14_01_0471_HE1</td>\n",
       "      <td>0.037599</td>\n",
       "      <td>949.822362</td>\n",
       "      <td>1.058824</td>\n",
       "      <td>0.003849</td>\n",
       "      <td>276.921322</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>14_01_0620_HE1</td>\n",
       "      <td>0.011149</td>\n",
       "      <td>1400.945465</td>\n",
       "      <td>1.715447</td>\n",
       "      <td>0.010651</td>\n",
       "      <td>276.467814</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>14_01_0627_HE1</td>\n",
       "      <td>0.005849</td>\n",
       "      <td>889.227088</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.008131</td>\n",
       "      <td>170.460120</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>14_01_0990_HE1</td>\n",
       "      <td>0.010565</td>\n",
       "      <td>1571.122442</td>\n",
       "      <td>1.323383</td>\n",
       "      <td>0.010255</td>\n",
       "      <td>328.508867</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          FileName  GanglionPerMM  GanglionArea  GanglioncellNum  \\\n",
       "0   14_01_0008_HE1       0.006488   1287.321894         1.405797   \n",
       "1   14_01_0293_HE1       0.008789   1460.626287         1.944954   \n",
       "2   14_01_0362_HE1       0.004426    465.815917         0.450000   \n",
       "3   14_01_0396_HE1       0.009005   1206.458839         1.045161   \n",
       "4   14_01_0413_HE1       0.012737   1228.463400         1.164179   \n",
       "5   14_01_0451_HE1       0.006948   1721.300619         1.000000   \n",
       "6   14_01_0470_HE1       0.002688   1129.185248         2.272727   \n",
       "7   14_01_0471_HE1       0.037599    949.822362         1.058824   \n",
       "8   14_01_0620_HE1       0.011149   1400.945465         1.715447   \n",
       "9   14_01_0627_HE1       0.005849    889.227088         1.000000   \n",
       "10  14_01_0990_HE1       0.010565   1571.122442         1.323383   \n",
       "\n",
       "    GanglioncellPerMM  GanglioncellArea  \n",
       "0            0.008269        288.188242  \n",
       "1            0.017752        186.015293  \n",
       "2            0.002088        182.457583  \n",
       "3            0.007722        326.828353  \n",
       "4            0.013570        146.078345  \n",
       "5            0.011653        208.471246  \n",
       "6            0.006868        180.986408  \n",
       "7            0.003849        276.921322  \n",
       "8            0.010651        276.467814  \n",
       "9            0.008131        170.460120  \n",
       "10           0.010255        328.508867  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "morpho_feature = pd.read_csv('/workspace/src/ganglion/CIPO_sm_info_231126.csv')\n",
    "data = morpho_feature.drop([10], axis=0).reset_index(drop=True)\n",
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# severity labeling\n",
    "\n",
    "threeG = [2, 2, 2, 2, 1, 0, 0, 1, 0, 2, 0]\n",
    "twoG = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]\n",
    "#print(len(threeG), len(twoG))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "standard_scaler = StandardScaler()\n",
    "minmax_scaler = MinMaxScaler()\n",
    "robust_scaler = RobustScaler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Two severity group"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Counter({1: 5, 0: 6})"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Counter(twoG)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(7, 5) (4, 5) (7,) (4,)\n"
     ]
    }
   ],
   "source": [
    "X_data = data.drop(['FileName'], axis=1)\n",
    "y_data = np.array(twoG)\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=4, random_state=42)\n",
    "print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "z_scaled_X_train = standard_scaler.fit_transform(X_train)\n",
    "z_scaled_X_test = standard_scaler.transform(X_test)\n",
    "\n",
    "mm_scaled_X_train = minmax_scaler.fit_transform(X_train)\n",
    "mm_scaled_X_test = minmax_scaler.transform(X_test)\n",
    "\n",
    "r_scaled_X_train = robust_scaler.fit_transform(X_train)\n",
    "r_scaled_X_test = robust_scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### **Feature correlation 확인** \n",
    "- 다중공선성 여부"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GanglionPerMM</th>\n",
       "      <th>GanglionArea</th>\n",
       "      <th>GanglioncellNum</th>\n",
       "      <th>GanglioncellPerMM</th>\n",
       "      <th>GanglioncellArea</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-0.452771</td>\n",
       "      <td>0.231305</td>\n",
       "      <td>0.202957</td>\n",
       "      <td>-0.215225</td>\n",
       "      <td>0.857446</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.197375</td>\n",
       "      <td>0.749916</td>\n",
       "      <td>1.314086</td>\n",
       "      <td>2.064708</td>\n",
       "      <td>-0.752219</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-0.681517</td>\n",
       "      <td>-2.227035</td>\n",
       "      <td>-1.766810</td>\n",
       "      <td>-1.701234</td>\n",
       "      <td>-0.808268</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.173381</td>\n",
       "      <td>-0.010676</td>\n",
       "      <td>-0.540264</td>\n",
       "      <td>-0.346886</td>\n",
       "      <td>1.466194</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.240774</td>\n",
       "      <td>0.055173</td>\n",
       "      <td>-0.294985</td>\n",
       "      <td>1.059288</td>\n",
       "      <td>-1.381398</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   GanglionPerMM  GanglionArea  GanglioncellNum  GanglioncellPerMM  \\\n",
       "0      -0.452771      0.231305         0.202957          -0.215225   \n",
       "1      -0.197375      0.749916         1.314086           2.064708   \n",
       "2      -0.681517     -2.227035        -1.766810          -1.701234   \n",
       "3      -0.173381     -0.010676        -0.540264          -0.346886   \n",
       "4       0.240774      0.055173        -0.294985           1.059288   \n",
       "\n",
       "   GanglioncellArea  \n",
       "0          0.857446  \n",
       "1         -0.752219  \n",
       "2         -0.808268  \n",
       "3          1.466194  \n",
       "4         -1.381398  "
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z_scaled_data = standard_scaler.fit_transform(X_data)\n",
    "df = pd.DataFrame(z_scaled_data, columns=['GanglionPerMM', 'GanglionArea', 'GanglioncellNum', 'GanglioncellPerMM', 'GanglioncellArea'])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 504x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "correlation_matrix = df.corr()\n",
    "\n",
    "# 히트맵 그리기\n",
    "plt.figure(figsize=(7,6))\n",
    "Hmap = sns.heatmap(correlation_matrix, annot=True, cmap=\"Blues\", fmt=\".2f\", linewidths=0.5)\n",
    "\n",
    "Hmap.set_xticklabels(Hmap.get_xticklabels(), rotation=30, ha='right', fontsize=10)\n",
    "Hmap.set_yticklabels(Hmap.get_yticklabels(), rotation=30, ha='right', fontsize=10)\n",
    "\n",
    "plt.title(\"Correlation Heatmap of Features\", fontsize=15)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. XGBoost"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2-1. Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def OneHot(test_y):\n",
    "    ohe = OneHotEncoder()\n",
    "    ohe_test_y = ohe.fit_transform(test_y.reshape(-1,1)).toarray()\n",
    "    return ohe_test_y\n",
    "\n",
    "def PredAndEval(train_X, train_y, test_X, test_y, classifier):\n",
    "    classifier.fit(train_X, train_y)\n",
    "    pred = classifier.predict(test_X)\n",
    "    prob = classifier.predict_proba(test_X)\n",
    "\n",
    "    ohe_test_y = OneHot(test_y)\n",
    "    acc = accuracy_score(test_y, pred)\n",
    "    auc = roc_auc_score(ohe_test_y, prob)\n",
    "    return pred, prob, acc, auc\n",
    "\n",
    "def ML_clf(train_X, train_y, test_X, test_y, params, classifier_type):\n",
    "    history = {'test_acc':[], 'test_auc':[], 'para':[]}\n",
    "\n",
    "    # 매개변수의 가능한 조합을 생성\n",
    "    param_combinations = product(*params.values())\n",
    "\n",
    "    if classifier_type == 'XGB':\n",
    "        for est, lr, col, gam, sub, lam in param_combinations:\n",
    "            clf = XGBClassifier(learning_rate=lr, n_estimators=est, colsample_bytree=col, \n",
    "                                gamma=gam, subsample=sub, reg_lambda=lam, random_state=42)\n",
    "                            \n",
    "            pred, prob, acc, auc = PredAndEval(train_X, train_y, test_X, test_y, clf)\n",
    "            para = [est, lr, col, gam, sub, lam]\n",
    "\n",
    "            [history[key].append(value) for key, value in zip(['test_acc', 'test_auc', 'para'], [acc, auc, para])]\n",
    "\n",
    "    elif classifier_type == 'RF':\n",
    "        for est, cri, depth, feat, sam in param_combinations:\n",
    "            clf = RandomForestClassifier(n_estimators=est, criterion=cri, max_depth=depth,\n",
    "                                         max_features=feat, max_samples=sam, bootstrap=True, random_state=42)\n",
    "                            \n",
    "            pred, prob, acc, auc = PredAndEval(train_X, train_y, test_X, test_y, clf)\n",
    "            para = [est, cri, depth, feat, sam]\n",
    "\n",
    "            [history[key].append(value) for key, value in zip(['test_acc', 'test_auc', 'para'], [acc, auc, para])]\n",
    "\n",
    "    return history"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def MLperScale(scaler_type, params, classifier_type):\n",
    "    global save_path\n",
    "    \n",
    "    if scaler_type == 'z':\n",
    "        hist = ML_clf(z_scaled_X_train, y_train, z_scaled_X_test, y_test, params, classifier_type)\n",
    "    elif scaler_type == 'mm':\n",
    "        hist = ML_clf(mm_scaled_X_train, y_train, mm_scaled_X_test, y_test, params, classifier_type)\n",
    "    elif scaler_type == 'r':\n",
    "        hist = ML_clf(r_scaled_X_train, y_train, r_scaled_X_test, y_test, params, classifier_type)\n",
    "    else:\n",
    "        print(f\"There is no {scaler_type} scaler type. Only put 'z', 'mm', or 'r' type.\")\n",
    "\n",
    "    with open(os.path.join(save_path, f\"{classifier_type}_{scaler_type}scale_2group_hist.pkl\"), \"wb\") as f:\n",
    "        pickle.dump(hist, f)\n",
    "        \n",
    "    return hist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "save_path = '/workspace/src/ganglion/stage4_ML_results/2group'\n",
    "\n",
    "if not os.path.exists(save_path):\n",
    "    os.mkdir(save_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "xgb_params = {\n",
    "    'n_estimators':[50, 70, 100, 200, 300, 400, 500], \n",
    "    'learning_rate':[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005], \n",
    "    'colsample_bytree':[0.3, 0.5, 0.7, 1.0],\n",
    "    'gamma':[0, 0.5, 1, 1.5],\n",
    "    'subsample':[0.3, 0.5, 0.7, 1.0],\n",
    "    'reg_lambda':[0, 1, 3, 5]\n",
    "}\n",
    "\n",
    "# Train\n",
    "#z_xgb_hist = MLperScale('z', xgb_params, \"XGB\")   # 이미 저장돼있음\n",
    "#mm_xgb_hist = MLperScale('mm', xgb_params, \"XGB\")\n",
    "#r_xgb_hist = MLperScale('r', xgb_params, \"XGB\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2-2. Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def GetHist(save_path, fname):\n",
    "    with open(os.path.join(save_path, fname), \"rb\") as f:\n",
    "        loaded_hist = pickle.load(f)\n",
    "    return loaded_hist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def BestCase(hist_pkl):\n",
    "    best_acc = np.max(hist_pkl['test_acc'])\n",
    "    best_auc = np.max(hist_pkl['test_auc'])\n",
    "    print(f\"Best Acc : {best_acc}\\nBest AUC : {best_auc}\")\n",
    "\n",
    "    best_acc_idx = hist_pkl['test_acc'].index(best_acc)\n",
    "    best_auc_idx = hist_pkl['test_auc'].index(best_auc)\n",
    "\n",
    "    best_acc_para, best_auc_para = hist_pkl['para'][best_acc_idx], hist_pkl['para'][best_auc_idx]\n",
    "    print(f\"Best Acc param : {best_acc_para}\\nBest AUC param : {best_auc_para}\")\n",
    "\n",
    "    return best_acc, best_auc, best_acc_para, best_auc_para"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "z_xgb_hist = GetHist(save_path, \"XGB_zscale_2group_hist.pkl\")\n",
    "mm_xgb_hist = GetHist(save_path, \"XGB_mmscale_2group_hist.pkl\")\n",
    "r_xgb_hist = GetHist(save_path, \"XGB_rscale_2group_hist.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "**Z-score\n",
      "Best Acc : 0.5\n",
      "Best AUC : 0.5\n",
      "Best Acc param : [50, 0.1, 0.3, 0, 0.3, 0]\n",
      "Best AUC param : [50, 0.1, 0.3, 0, 0.3, 0]\n",
      "\n",
      "**Min-Max\n",
      "Best Acc : 0.5\n",
      "Best AUC : 0.5\n",
      "Best Acc param : [50, 0.1, 0.3, 0, 0.3, 0]\n",
      "Best AUC param : [50, 0.1, 0.3, 0, 0.3, 0]\n",
      "\n",
      "**Robust\n",
      "Best Acc : 0.5\n",
      "Best AUC : 0.5\n",
      "Best Acc param : [50, 0.1, 0.3, 0, 0.3, 0]\n",
      "Best AUC param : [50, 0.1, 0.3, 0, 0.3, 0]\n"
     ]
    }
   ],
   "source": [
    "print(\"\\n**Z-score\")\n",
    "_, _, z_xgb_acc_para, z_xgb_auc_para = BestCase(z_xgb_hist) \n",
    "\n",
    "print(\"\\n**Min-Max\")\n",
    "_, _, mm_xgb_acc_para, mm_xgb_auc_para = BestCase(mm_xgb_hist) \n",
    "\n",
    "print(\"\\n**Robust\")\n",
    "_, _, r_xgb_acc_para, r_xgb_auc_para = BestCase(r_xgb_hist) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Final test acc & auc : 0.5, 0.5\n",
      "[[0 2]\n",
      " [0 2]]\n"
     ]
    }
   ],
   "source": [
    "# BEST ACC 기준\n",
    "est, lr, col, gam, sub, lam = z_xgb_acc_para\n",
    "xgb_clf = XGBClassifier(n_estimators=est, learning_rate=lr, colsample_bytree=col, \n",
    "                       gamma=gam, subsample=sub, reg_lambda=lam, random_state=42)\n",
    "\n",
    "pred, prob, acc, auc = PredAndEval(r_scaled_X_train, y_train, r_scaled_X_test, y_test, xgb_clf)\n",
    "cm = confusion_matrix(y_test, pred)\n",
    "print(f\"Final test acc & auc : {round(acc, 5)}, {round(auc, 5)}\")\n",
    "print(cm)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# # BEST AUC 기준\n",
    "# est, lr, col, gam, sub, lam = best_auc_para\n",
    "# xgb_clf = XGBClassifier(n_estimators=est, learning_rate=lr, colsample_bytree=col, \n",
    "#                        gamma=gam, subsample=sub, reg_lambda=lam, random_state=42)\n",
    "\n",
    "# pred, prob, acc, auc = PredAndEval(z_scaled_X_train, y_train, z_scaled_X_test, y_test, xgb_clf)\n",
    "# cm = confusion_matrix(y_test, pred)\n",
    "# print(f\"Final test acc & auc : {round(acc, 5)}, {round(auc, 5)}\")\n",
    "# print(cm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Random forest"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3-1. Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_params = {\n",
    "    'n_estimators':[30, 50, 70, 100, 200, 300, 400, 500, 600], \n",
    "    'criterion':['gini', 'entropy'],  \n",
    "    'max_depth':[1, 3, 5, 7, 9, 11, 15],\n",
    "    'max_features':['sqrt', None, 0.5, 0.7],\n",
    "    'max_samples':[0.3, 0.5, 0.7, 1.0]\n",
    "}\n",
    "\n",
    "# Train\n",
    "# z_rf_hist = MLperScale('z', rf_params, \"RF\")\n",
    "# mm_rf_hist = MLperScale('mm', rf_params, \"RF\")\n",
    "# r_rf_hist = MLperScale('r', rf_params, \"RF\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3-2. Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "z_rf_hist = GetHist(save_path, \"RF_zscale_2group_hist.pkl\")\n",
    "mm_rf_hist = GetHist(save_path, \"RF_mmscale_2group_hist.pkl\")\n",
    "r_rf_hist = GetHist(save_path, \"RF_rscale_2group_hist.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "**Z-score\n",
      "Best Acc : 0.75\n",
      "Best AUC : 0.75\n",
      "Best Acc param : [30, 'gini', 1, None, 0.7]\n",
      "Best AUC param : [30, 'gini', 1, 'sqrt', 0.5]\n",
      "\n",
      "**Min-Max\n",
      "Best Acc : 0.75\n",
      "Best AUC : 0.75\n",
      "Best Acc param : [30, 'gini', 1, None, 0.7]\n",
      "Best AUC param : [30, 'gini', 1, 'sqrt', 0.5]\n",
      "\n",
      "**Robust\n",
      "Best Acc : 0.75\n",
      "Best AUC : 0.75\n",
      "Best Acc param : [30, 'gini', 1, None, 0.7]\n",
      "Best AUC param : [30, 'gini', 1, 'sqrt', 0.5]\n"
     ]
    }
   ],
   "source": [
    "print(\"\\n**Z-score\")\n",
    "_, _, z_rf_acc_para, z_rf_auc_para = BestCase(z_rf_hist) \n",
    "\n",
    "print(\"\\n**Min-Max\")\n",
    "_, _, mm_rf_acc_para, mm_rf_auc_para = BestCase(mm_rf_hist) \n",
    "\n",
    "print(\"\\n**Robust\")\n",
    "_, _, r_rf_acc_para, r_rf_auc_para = BestCase(r_rf_hist) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Final test acc & auc : 0.75, 0.75\n",
      "[[1 1]\n",
      " [0 2]]\n"
     ]
    }
   ],
   "source": [
    "# BEST ACC 기준\n",
    "est, cri, depth, feat, sam = z_rf_acc_para\n",
    "rf_clf = RandomForestClassifier(n_estimators=est, criterion=cri, max_depth=depth,\n",
    "                                max_features=feat, max_samples=sam, bootstrap=True, random_state=42)\n",
    "\n",
    "pred, prob, acc, auc = PredAndEval(r_scaled_X_train, y_train, r_scaled_X_test, y_test, rf_clf)\n",
    "cm = confusion_matrix(y_test, pred)\n",
    "print(f\"Final test acc & auc : {round(acc, 5)}, {round(auc, 5)}\")\n",
    "print(cm)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True : [0 1 1 0]\n",
      "Pred : [1 1 1 0]\n",
      "    GanglionPerMM  GanglionArea  GanglioncellNum  GanglioncellPerMM  \\\n",
      "5        0.006948   1721.300619         1.000000           0.011653   \n",
      "0        0.006488   1287.321894         1.405797           0.008269   \n",
      "9        0.005849    889.227088         1.000000           0.008131   \n",
      "10       0.010565   1571.122442         1.323383           0.010255   \n",
      "\n",
      "    GanglioncellArea  \n",
      "5         208.471246  \n",
      "0         288.188242  \n",
      "9         170.460120  \n",
      "10        328.508867  \n"
     ]
    }
   ],
   "source": [
    "print(f\"True : {y_test}\")\n",
    "print(f\"Pred : {pred}\")\n",
    "print(X_test)\n",
    "# test set은 순서대로 14_01_0451_HE1, 14_01_0008_HE1, 14_01_0627_HE1, 14_01_0990_HE1\n",
    "# 14_01_0451_HE1 틀림"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### SHAP 확인"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_train_explainer = shap.TreeExplainer(rf_clf, data=r_scaled_X_train, model_output='probability')\n",
    "rf_shap_values_train = rf_train_explainer.shap_values(r_scaled_X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_test_explainer = shap.TreeExplainer(rf_clf, data=r_scaled_X_test, model_output='probability')\n",
    "rf_shap_values_test = rf_test_explainer.shap_values(r_scaled_X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "features = ['GanglionPerMM', 'GanglionArea', 'GanglioncellNum', 'GanglioncellPerMM', 'GanglioncellArea']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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X7G8nNXrWutJKJvK6wRgJ3dfkNd75231Ac67FccZ8EZjA5HdfR/gFGB4RLzu7scAX69Z6UVEREqm03ZRRcR84HbgGnf/WEVS7/y9L7AgBzcHUIylaemcSym6d44GcPfPUHQ7teQ+4ER3N3fvAxwF3NvcAe7+CeBzwICI6B8R/YF+RZJv28ghzwP/5+5frTjH37n7xhR1XQK86e49KYIkERERaUKnDXCy0RTdM7PcfZ67PwzsTjE4dwzFuJs5wBFANV1NAMcBp7v7XOBMYC5F8NCccylaWOYCjwLXR8TdLRxzPHBXRLxTvyEi3gd+C3yt4c4RUQeMAo5y96fcfR7wU4ouqbuBP1J0Sz1EMf5IREREmmAp1bq3pnPLT0S9l1t+dgJmAoPK+ESSTajrWh9uls46vKOL0DHS1I4ugYhIR7DGNnbmMThtZThwibvXX5CTyxjciIiIdGVdLsCJiBnAjI4uh4iIiLSdzj4GR0RERKTVFOCIiIhI6SjAERERkdLpck9RdSXTpk1Lo0aN6uhiiIiItKVGn6JSC46IiIiUjgIcERERKR0FOCIiIlI6CnBERESkdBTgiIiISOkowBEREZHSUYAjIiIipaMAR0REREpHAY6IiIiUjt5kXGI2oa7LfLjprMM7ugjtJ03t6BKIyBqyCXVtev50Zvean/Pll19mp512YsmSJXTr1o399tuPY489lpNOOukj+y5YsIABAwawbNkyunevfVma0OibjNstdxEREVn3bLvttrz77rsdXYxWUxeViIiIlI4CHBERkS6of//+XHLJJQwePJjevXtz4oknsnjxYkaOHEmfPn0YMWIEb731FgsWLMDMqKv7aPfa8uXLOfPMM9l8883ZbrvtuPPOOzugJo1TgCMiItJF/frXv+bee+/lhRdeYNq0aYwcOZILLriA119/nRUrVnD55Zc3e/w111zD9OnTmT17NhHBbbfd1k4lb5kCHBERkS7q29/+NltuuSVbb701++yzD3vuuSdDhw6lV69eHHroocyePbvZ42+55RZOP/10+vXrx6abbsoPfvCDdip5yxTgiIiIdFFbbrnlyuUNNtjgI+stDS5etGgR/fr1W7n+qU99qvaFXEM1f4rK3dcHzgaOBury13xgbEQ80wb5zQQmRMR0d/8xMC8ibq51Pq0owyQgIuLKivSLgO8C20TEX9qzbCIiIm3lk5/8JH/+859Xrr/88ssdWJrVtUULzrXAYGDPiNgZGJK3DWqDvFYTEWPbO7hpibt3A74KPJy/N7WfHtkXEZF1yhFHHMHll1/OK6+8wltvvcX48eM7ukgr1fSPqrsPBA6laKl4GyAiEnBnTt8fOA/olfM+PyJuymkzgVnAZ4GtgFsiYkxO24kiSOoNzAF2AM6LiOkN8p9Ebj1x942AK4BhOXlyRFxcRV5bA5cDA/NxUyLiQnffGLiUInjrBTwInBERy1u4LAcCfwTGAj8H/r2ivAk4BzgIuNvdL2kqD3f/HnBUvm7vA6dExJwW8hYRkU6kLV7E15FOPvlkXnjhBXbbbTc23nhjzjzzTB544IGOLhZQ+xacocD8iHirifQngL0jYigwApjg7ptUpG8L7JvPc1IOmACuB66IiF2Ay1gVtDTnRxT12xUYDhzv7iOryOtXwGMRMTgiBgPX5O2XAg9FxB4UrVJbACdUUY4TgGsj4mGgh7vv2SD9bxExLCJ+1EIek/N+Q3Pdfl5F3iIiIo1asGABI0aMWLn+q1/9inHjxq1cP+mkk7jvvvvo378/KaWVbyaeOXPmyrcYd+/enZ/85Ce8+eab/OlPf+LUU09dbd+O1KYlyC0vNwIbAncBVwETczBRB2xK0XX1WD7k1ohYASxx92eB7d19MbBLPg8REe7+VBXZjwC+k1uQlrr7lLztrmby+l+KYOiA+pNExBt58WBgj9ySQq7TKy3UfwtgP+C4vOk6ioDl8YrdrqtYbi6P3d39Xymu2Qrg081XX0REpOuqdYAzGxjo7n0j4u08qHiIu58GOPAz4A7gsIhI7v4CRVdMvfcrlpc3KF+t51VqLq/GGHBIRLzUijy+CqwPzHV3ch693f30iPhb3qdyiHqjebh7D+A2YN+IeMLdtwJebUU5REREupSadlFFxHzgduAad/9YRVLv/L0vsCAHNwdQjKVp6ZxLgXkUT2Xh7p+h6HZqyX3Aie5u7t6HYvzKvS3k9S7w3xRPPJHz2zwv3gGMyYOGcffN3X1AC2X4GkXA0j9/bQP8D9DUzJBN5VE/Zql+qPq3WshXRESkS2uLp6hGA88Bs9x9nrs/DOxOMXB3DMW4mznAEUA1XU1QdPGc7u5zgTOBucCSFo45l6JFZC7wKHB9RNxdRV7HAnu5+9Pu/iRwYt5+OkVLz5O5HHcDWzd1kjzWZlOg4WirG2h67E6jeeQgbyzFNf0D8F4V9RAREemyLKVa9/zUXn4i6r3c8rMTMBMY1MxgZgFsQl3n/3BrJJ3VVKNYCaWpHV0CEZHOxBrb2PHDnKszHLjE3esrcbKCGxEREWnKOhHgRMQMYEZHl0NERETWDZqLSkREREpnnWjBERERKQU7pG3PrzF6K6kFR0REREpHLTgldseguxg1alRHF6N9nDm1o0sgIiJVWr58Od26dWvTPNSCIyIi0gVddNFFbL311vTp04dBgwZx//33s2LFCsaPH8/222/PZpttxhFHHMFf//pXAEaOHMmVV1652jl22203fvOb3wDw3HPPccABB7DpppsyaNAgbrnllpX7jR49mlNOOYUDDzyQ3r178+CDD7Jo0SK+/OUv8/GPf5wBAwZw+eWX17R+CnBERES6mOeff54rr7ySWbNm8c4773DPPffQv39/rrjiCqZOncpDDz3EokWL2GSTTTj11FMBOProo5kyZcrKczzzzDMsXLiQgw46iPfee48DDjiAY445hr/85S/cdNNNfOtb3+KZZ55Zuf+NN97I2WefzTvvvMPw4cMZNWoUu+22G6+++ir3338/l112Gffcc0/N6qgAR0REpIvp1q0bH3zwAc888wzLli2jf//+bL/99vz85z/n/PPPZ5tttqFnz56MGzeO2267jbq6Og499FDmzJnDwoULAbjhhhs47LDD6NmzJ9OnT6d///587Wtfo3v37gwdOpQvf/nL3HrrrSvz/NKXvsRee+3Feuutx9y5c3n99dcZO3YsPXr0YLvttuPkk0/mpptuqlkdFeCIiIh0MTvssAOXXXYZ48aNY4sttuCoo45i0aJFLFy4kEMPPZS+ffvSt29fdtxxR7p168bixYvp06cPBx100MogZMqUKXzlK18BYOHChTz++OMrj+vbty833HADr7322so8+/Xrt3J54cKFLFq0aLX9L7jgAhYvXlyzOmqQsYiISBd0zDHHcMwxx7B06VK+8Y1v8P3vf59+/foxceJE9tprr0aPOfrooznnnHPYd999ef/99/n85z8PFMHL5z73Oe69t+k5rc1WzajQr18/BgwYwPz582tbqQpqwREREelinn/+eR544AE++OADevXqxQYbbMB6663HN7/5Tc4+++yV3VCvv/46t99++8rjDjzwQBYuXMjYsWM5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",
      "text/plain": [
       "<Figure size 576x252 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "shap.summary_plot(rf_shap_values_train, r_scaled_X_train, features, class_names=['mild', 'severe'],\n",
    "                  class_inds='original', max_display=10, show=False)\n",
    "\n",
    "fig, ax = plt.gcf(), plt.gca()\n",
    "\n",
    "ax.tick_params(labelsize=11)\n",
    "ax.set_xlabel(\"mean(|SHAP value|)\\n(average impact on model output magnitude)\", fontsize=11)\n",
    "\n",
    "plt.tight_layout()\n",
    "#plt.show()\n",
    "plt.savefig('/workspace/src/ganglion/stage4_ML_results/2group/rf_train_SHAP.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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6+7ekyJJEhERkWZ02AQnG03RPTPT3ee6+8PAbhSDc8dSjLuZDRwGVNPVBHAMMMbd5wCnAXMokoeWnEPRwjIHeBS4PiLubuWcY4G7IuLthg0R8R7wa+ArjQ+OiOXAKOAId3/K3ecCP6bokrob+ANFt9RDFOOPREREpBmWUq17azq2/ETUu7nlZyAwAxhQxieSbMLyrvXhrgPp9EPbO4RySVPaOwIR6fysqY0deQxOvQwDLnb3hgo5sYzJjYiISFfW5RKciJgOTG/vOERERKR+OvoYHBEREZE2U4IjIiIipaMER0REREqnyz1F1ZVMnTo1jRo1qr3DEBERqacmn6JSC46IiIiUjhIcERERKR0lOCIiIlI6SnBERESkdJTgiIiISOkowREREZHSUYIjIiIipaMER0REREpHCY6IiIiUjt5kXGI2YXnpP9x0+qHtHUJ9pSntHYGI1JBNWF7X66fTutf8mi+++CIDBw5kyZIldOvWjX333Zejjz6aE0444X3HLliwgG233ZZly5bRvXvtY2lGk28yXmeli4iISOezzTbb8M4777R3GG2mLioREREpHSU4IiIiXVD//v25+OKLGTRoEL179+b4449n8eLFjBw5kj59+jB8+HDeeOMNFixYgJmxfPn7u9dWrFjBaaedxuabb852223HnXfe2Q530jQlOCIiIl3UL3/5S+69916ef/55pk6dysiRIzn//PN59dVXWblyJZdddlmL51999dVMmzaNWbNmERHcdttt6yjy1inBERER6aK++c1v8pGPfIStttqKvffemz322IMhQ4bQq1cvDj74YGbNmtXi+bfccgtjxoyhX79+bLrppnzve99bR5G3TgmOiIhIF/WRj3xk1fIGG2zwvvXWBhcvWrSIfv36rVr/+Mc/XvsgP6CqnqJy9/WBM4AjgeX5Zz4wLiLm1Tood58BTIiIae7+Q2BuRNxc63LaEMMkICLiCncfDVwKLAB6AM8AJ0bEX9t4/UnAscDOETE3b9sW+APwq4g41N37A38EbouIf23i3F0i4ukPfJMiIiJr4WMf+xh/+tOfVq2/+OKL7RjNmqptwbkGGATsERE7AbvmbQPqFNcqETFuXSc3VbgvInYFdgYS8IO2nOzu3fLiExSJSoPRQOP2wL8Cu7j7JvncjYC9gZfbHLWIiEgNHXbYYVx22WW89NJLvPHGG4wfP769Q1ql1RYcd98ROBjYOiLeBIiIBNyZ9+8PnAv0ytc7LyJuyvtmADOBTwNbArdExNi8byBFktQbmA3sAJwbEdMalT+J1a0nGwGXA0Pz7usi4qIqytoKuAzYMZ83OSIucPeNgUsokrdewIPAqRGxotWaK4Ja6e4PAAfmco4FvpHrYQlwUkQ8l1t9jgbezjEcnS9xK3Ccu38PWAkcAVyV76FBAm6maD37MfCvwBTgi9XEKCIiHUc9XsTXnk488USef/55Bg8ezMYbb8xpp53GAw880N5hAdW14AwB5kfEG83sfwLYKyKGAMOBCQ2tDdk2wD75OifkhAngeuDyiNiZostnKK07M8e8CzAMONbdR1ZR1i+AxyJiUEQMAq7O2y8BHoqI3SlapbYAjqsiDgDcvSdwEDDL3fcGDgP2iYjdgIuBiRWH/yNwWkTsHBGz87Z3gEeBzwL7Ak8DrzdR1LXAMXn5WGBStTGKiIg0ZcGCBQwfPnzV+i9+8QvOOuusVesnnHAC9913H/379yeltOrNxDNmzFj1FuPu3bvzox/9iNdff50//vGPnHzyyWsc257aHEFuebkR2BC4C7gSmJiTieXAphRdV4/lU26NiJXAEnd/Btje3RdTdO/cCEXzjLs/VUXxw4Fv5Rakt9x9ct52Vwtl/ZkiGRrRcJGIeC0vHgTs7u7fyesbAi9VE4e7z87LjwAXAP8ODAYed3coXh1dmeg9HBF/aOJak4CvAUvz8maND4iIF9z9PXc/AOgdEXNyGSIiItKEahKcWcCO7t43It7Mg4p3dfdTAAd+AtwBHBIRyd2fp+juafBexfKKRmXWeq6klspqigFfiIgX2ljOfRGxxiRI7m7AxIgY18w5zQ1Fn0HR9dQDOJ7V3VeNXQtcB5zdxlhFRES6nFa7qCJiPnA7cLW7f6hiV+/8Z19gQU5uRlCMpWntmm8BcynGleDun6LodmrNfcDx7m7u3odizMq9rZT1DvA/wLcbtrn75nnxDmBsw6Bfd988P8n0QUwFjnH3rfO1urn7bq2dlFujvg18OyJamoXtVmACcMMHjE9ERKTLqLaLajTF+JeZ7r4MeANYBIyn6FL5sbufTTHIt5quJijGlEzMA2zn5J8lrZxzDnBFPhbg+oi4u4qyjgauzIOAV1B0jV0IjAEuAp5090TRTTSG4tHsNomI37r7GcAdOWHqQZGU/K6Kc1u9h5yodZzh6SIiIh2YpVTrXqLq5Cei3s0tPwMpumoGtDCYWdrIJixvnw93HUqnH9r6QZ1ZmtLeEYiIdHTW1Mb2HOY8DLg4j12B4mV5Sm5ERERkrbVbghMR04Hp7VW+iIiIlJfmohIREZHSaf838YiIiHQV9oX6Xl/j9lZRC46IiIiUjlpwSuyOAXcxatSo9g6jvk6b0t4RiIhIG61YsYJu3bq1fuBaUAuOiIhIF3ThhRey1VZb0adPHwYMGMD999/PypUrGT9+PNtvvz2bbbYZhx12GH/9618BGDlyJFdcccUa1xg8eDC/+tWvAHj22WcZMWIEm266KQMGDOCWW25Zddzo0aM56aSTOOCAA+jduzcPPvggixYt4otf/CIf/vCH2Xbbbbnssstqen9KcERERLqY5557jiuuuIKZM2fy9ttvc88999C/f38uv/xypkyZwkMPPcSiRYvYZJNNOPnkkwE48sgjmTx58qprzJs3j4ULF3LggQfy7rvvMmLECI466ij+8pe/cNNNN/GNb3yDefPmrTr+xhtv5IwzzuDtt99m2LBhjBo1isGDB/Pyyy9z//33c+mll3LPPffU7B6V4IiIiHQx3bp1Y+nSpcybN49ly5bRv39/tt9+e376059y3nnnsfXWW9OzZ0/OOussbrvtNpYvX87BBx/M7NmzWbhwIQA33HADhxxyCD179mTatGn079+fr3zlK3Tv3p0hQ4bwxS9+kVtvvXVVmZ///OfZc889WW+99ZgzZw6vvvoq48aNo0ePHmy33XaceOKJ3HTTTTW7RyU4IiIiXcwOO+zApZdeyllnncUWW2zBEUccwaJFi1i4cCEHH3wwffv2pW/fvnzyk5+kW7duLF68mD59+nDggQeuSkImT57Ml770JQAWLlzI448/vuq8vn37csMNN/DKK6+sKrNfv36rlhcuXMiiRYvWOP78889n8eLFNbtHDTIWERHpgo466iiOOuoo3nrrLb72ta/x3e9+l379+jFx4kT23HPPJs858sgjOfvss9lnn31477332G+//YAiefnMZz7Dvfc2P/+12eoZFfr168e2227L/Pnza3tTFdSCIyIi0sU899xzPPDAAyxdupRevXqxwQYbsN566/H1r3+dM844Y1U31Kuvvsrtt9++6rwDDjiAhQsXMm7cOA4//HDWW69I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",
      "text/plain": [
       "<Figure size 576x252 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "shap.summary_plot(rf_shap_values_test, r_scaled_X_test, features, class_names=['mild', 'severe'],\n",
    "                  class_inds='original', max_display=10, show=False)\n",
    "\n",
    "fig, ax = plt.gcf(), plt.gca()\n",
    "\n",
    "ax.tick_params(labelsize=11)\n",
    "ax.set_xlabel(\"mean(|SHAP value|)\\n(average impact on model output magnitude)\", fontsize=11)\n",
    "\n",
    "plt.tight_layout()\n",
    "#plt.show()\n",
    "plt.savefig('/workspace/src/ganglion/stage4_ML_results/2group/rf_test_SHAP.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 720x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 720x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "for i, cls_name in enumerate(['mild', 'severe']):\n",
    "    # train\n",
    "    plt.figure()\n",
    "    shap.summary_plot(rf_shap_values_train[i], r_scaled_X_train, features, max_display=10, plot_size=(10,6), show=False)\n",
    "    plt.tight_layout()\n",
    "    #plt.show()\n",
    "    plt.savefig('/workspace/src/ganglion/stage4_ML_results/2group/rf_train_' + str(cls_name) + '_beeswarm.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 720x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 720x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "for i, cls_name in enumerate(['mild', 'severe']):\n",
    "    # test\n",
    "    plt.figure()\n",
    "    shap.summary_plot(rf_shap_values_test[i], r_scaled_X_test, features, max_display=10, plot_size=(10,6), show=False)\n",
    "    plt.tight_layout()\n",
    "    #plt.show()\n",
    "    plt.savefig('/workspace/src/ganglion/stage4_ML_results/2group/rf_test_' + str(cls_name) + '_beeswarm.png')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### permutation importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GanglionArea: -0.05714285714285716\n",
      "GanglioncellPerMM: -0.05714285714285716\n",
      "GanglioncellNum: -0.04285714285714287\n",
      "GanglioncellArea: -0.028571428571428602\n",
      "GanglionPerMM: 0.1285714285714285\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "r = permutation_importance(rf_clf, r_scaled_X_train, y_train, n_repeats=10, random_state=42)\n",
    "\n",
    "sorted_idx = r.importances_mean.argsort()\n",
    "\n",
    "for index in sorted_idx:\n",
    "    print(f\"{X_train.columns[index]}: {r.importances_mean[index]}\")\n",
    "\n",
    "plt.barh(range(len(r.importances_mean)), r.importances_mean[sorted_idx])\n",
    "plt.yticks(range(len(r.importances_mean)), X_train.columns[sorted_idx])  # y축 눈금 설정\n",
    "plt.ylabel(\"Feature\")\n",
    "plt.xlabel(\"Importance\")\n",
    "plt.title(\"Feature Importance\")\n",
    "plt.show()\n",
    "\n",
    "#############################\n",
    "# permutation importance에서 음수값은 해당 feature가 permute 되었을 때, 오히려 성능이 올라갔다는 것 \n",
    "# (해당 feature을 permute 했을 때, 성능의 증감을 통해서 중요도를 나열하고, 따라서 성능이 많이 떨어진 경우, 해당 feature가 모델 학습에 중요하다는 것)\n",
    "# 즉, 음수 값을 가지는 변수들은 모델에 영향을 끼치지 못하는 중요하지 않은 feature\n",
    "# 워낙 작은 dattset이라 더 그러함.. \n",
    "#############################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GanglionPerMM: 0.0\n",
      "GanglioncellNum: 0.0\n",
      "GanglioncellArea: 0.0\n",
      "GanglionArea: 0.15\n",
      "GanglioncellPerMM: 0.15\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# test\n",
    "r = permutation_importance(rf_clf, r_scaled_X_test, y_test, n_repeats=10, random_state=42)\n",
    "\n",
    "sorted_idx = r.importances_mean.argsort()\n",
    "\n",
    "for index in sorted_idx:\n",
    "    print(f\"{X_train.columns[index]}: {r.importances_mean[index]}\")\n",
    "\n",
    "plt.barh(range(len(r.importances_mean)), r.importances_mean[sorted_idx])\n",
    "plt.yticks(range(len(r.importances_mean)), X_train.columns[sorted_idx])  # y축 눈금 설정\n",
    "plt.ylabel(\"Feature\")\n",
    "plt.xlabel(\"Importance\")\n",
    "plt.title(\"Feature Importance\")\n",
    "plt.show()\n",
    "\n",
    "#############################\n",
    "# permutation importance에서 음수값은 해당 feature가 permute 되었을 때, 오히려 성능이 올라갔다는 것 \n",
    "# (해당 feature을 permute 했을 때, 성능의 증감을 통해서 중요도를 나열하고, 따라서 성능이 많이 떨어진 경우, 해당 feature가 모델 학습에 중요하다는 것)\n",
    "# 즉, 음수 값을 가지는 변수들은 모델에 영향을 끼치지 못하는 중요하지 않은 feature\n",
    "# 워낙 작은 dattset이라 더 그러함.. \n",
    "#############################"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### feature importance \n",
    "\n",
    "train set에 대한 것이라고 함"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GanglionArea: 0.07142857142857142\n",
      "GanglioncellPerMM: 0.14285714285714285\n",
      "GanglioncellArea: 0.17857142857142858\n",
      "GanglioncellNum: 0.25\n",
      "GanglionPerMM: 0.35714285714285715\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 특성 중요도 확인\n",
    "feature_importance = rf_clf.feature_importances_\n",
    "\n",
    "# 중요도를 기준으로 내림차순으로 특성의 인덱스 정렬\n",
    "sorted_idx = np.argsort(feature_importance)\n",
    "\n",
    "# 각 특성의 중요도와 이름 출력\n",
    "for index in sorted_idx:\n",
    "    print(f\"{X_train.columns[index]}: {feature_importance[index]}\")\n",
    "\n",
    "# 중요도 시각화\n",
    "plt.barh(range(len(feature_importance)), feature_importance[sorted_idx])\n",
    "plt.yticks(range(len(feature_importance)), X_train.columns[sorted_idx])  # y축 눈금 설정\n",
    "plt.ylabel(\"Feature\")\n",
    "plt.xlabel(\"Importance\")\n",
    "plt.title(\"Feature Importance\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Final test acc & auc : 0.5, 0.75\n",
      "[[1 1]\n",
      " [1 1]]\n"
     ]
    }
   ],
   "source": [
    "# BEST AUC 기준\n",
    "est, cri, depth, feat, sam = r_rf_auc_para\n",
    "rf_clf = RandomForestClassifier(n_estimators=est, criterion=cri, max_depth=depth,\n",
    "                                max_features=feat, max_samples=sam, bootstrap=True, random_state=42)\n",
    "\n",
    "pred, prob, acc, auc = PredAndEval(r_scaled_X_train, y_train, r_scaled_X_test, y_test, rf_clf)\n",
    "cm = confusion_matrix(y_test, pred)\n",
    "print(f\"Final test acc & auc : {round(acc, 5)}, {round(auc, 5)}\")\n",
    "print(cm)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GanglioncellPerMM: 0.15384615384615385\n",
      "GanglioncellArea: 0.15384615384615385\n",
      "GanglionPerMM: 0.19230769230769232\n",
      "GanglionArea: 0.19230769230769232\n",
      "GanglioncellNum: 0.3076923076923077\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 특성 중요도 확인\n",
    "feature_importance = rf_clf.feature_importances_\n",
    "\n",
    "# 중요도를 기준으로 내림차순으로 특성의 인덱스 정렬\n",
    "sorted_idx = np.argsort(feature_importance)\n",
    "\n",
    "# 각 특성의 중요도와 이름 출력\n",
    "for index in sorted_idx:\n",
    "    print(f\"{X_train.columns[index]}: {feature_importance[index]}\")\n",
    "\n",
    "# 중요도 시각화\n",
    "plt.barh(range(len(feature_importance)), feature_importance[sorted_idx])\n",
    "plt.yticks(range(len(feature_importance)), X_train.columns[sorted_idx])  # y축 눈금 설정\n",
    "plt.ylabel(\"Feature\")\n",
    "plt.xlabel(\"Importance\")\n",
    "plt.title(\"Feature Importance\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.12"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
