#!/usr/bin/env python
# coding: utf-8

# ### XGB vs ESMM 模型评估对比：
# 
# #### 口径：
# * 客户：`预测时间节点`在**合作中，非特行、X客户和框架客户**5个月后还没到期的金额（如果到期时间在5个月内则为0）
# * 时间：2024年
# #### 评估指标：
# 预估-paylastyear=新付
# * Y1: 预估值和实际值的MAPE。大金额（3w+）整体MAPE，小金额（3w-）MAPE分段统计
# * Y2: 预测客户能续大在实际续大中的占比，续大为1，0/1二分类问题
# * Y3：定义续约边界，看预测续约概率高的客户在实际续约客户中的占比，续约边界：预测值大于客户去年价值的80%

# In[28]:


import pandas as pd
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',    200)
pd.set_option('display.max_columns', 200)

df = pd.read_csv('xgb&esmm.csv')
df = df[df.pay_last_year_bucket != 'else']
df = df.fillna(0)
df.head(5)


# In[4]:


# Y1看 = 预估多少钱、实际回多少钱的MAPE。大数MAPE，小数对MAPE分段做个统计，0-10%误差、10-20%误差这种。


# In[32]:


# 先过滤掉实际为 0 的客户，避免除0
df_valid = df[df['value_after0_least5'] > 0].copy()

# 计算 MAPE
df_valid['mape_xgb'] = abs(df_valid['v2_pred_value'] - df_valid['value_after0_least5']) / df_valid['value_after0_least5']
df_valid['mape_esmm'] = abs(df_valid['pred_value'] - df_valid['value_after0_least5']) / df_valid['value_after0_least5']

# 大数（高价值客户：pay_last_year_bucket 为 '3w+'）
df_big = df_valid[df_valid['pay_last_year_bucket'] == '3w+']

print("高价值客户（3w+）整体 MAPE：")
print("XGB模型:", df_big['mape_xgb'].mean())
print("ESMM模型:", df_big['mape_esmm'].mean())

# 小数（中低价值客户：pay_last_year_bucket != '3w+'）进行 MAPE 分段
df_small = df_valid[df_valid['pay_last_year_bucket'] != '3w+']

bins = [0, 0.1, 0.2, 0.3, 0.5, 1, float('inf')]
labels = ['0~10%', '10~20%', '20~30%', '30~50%', '50~100%', '>100%']

df_small['mape_bin_xgb'] = pd.cut(df_small['mape_xgb'], bins=bins, labels=labels)
df_small['mape_bin_esmm'] = pd.cut(df_small['mape_esmm'], bins=bins, labels=labels)

print("中低价值客户（小数）XGB模型 MAPE分布：")
print(df_small['mape_bin_xgb'].value_counts(normalize=True).sort_index())

print("中低价值客户（小数）ESMM模型 MAPE分布：")
print(df_small['mape_bin_esmm'].value_counts(normalize=True).sort_index())


# In[36]:


# 高价值客户整体 MAPE 对比图

df_big = df[(df['value_after0_least5'] > 0) & (df['pay_last_year_bucket'] == '3w+')].copy()
df_big['mape_xgb'] = abs(df_big['v2_pred_value'] - df_big['value_after0_least5']) / df_big['value_after0_least5']
df_big['mape_esmm'] = abs(df_big['pred_value'] - df_big['value_after0_least5']) / df_big['value_after0_least5']

mape_mean_big = {
    'XGB': df_big['mape_xgb'].mean(),
    'ESMM': df_big['mape_esmm'].mean()
}

plt.figure(figsize=(6, 4))
plt.bar(mape_mean_big.keys(), mape_mean_big.values(), color=['skyblue', 'salmon'])
plt.ylabel('MAPE')
plt.title('3w+ MAPE')
plt.ylim(0, max(mape_mean_big.values()) * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


# 中低价值客户分段误差图

df_small = df[(df['value_after0_least5'] > 0) & (df['pay_last_year_bucket'] != '3w+')].copy()
df_small['mape_xgb'] = abs(df_small['v2_pred_value'] - df_small['value_after0_least5']) / df_small['value_after0_least5']
df_small['mape_esmm'] = abs(df_small['pred_value'] - df_small['value_after0_least5']) / df_small['value_after0_least5']

bins = [0, 0.1, 0.2, 0.3, 0.5, 1, float('inf')]
labels = ['0~10%', '10~20%', '20~30%', '30~50%', '50~100%', '>100%']

df_small['mape_bin_xgb'] = pd.cut(df_small['mape_xgb'], bins=bins, labels=labels)
df_small['mape_bin_esmm'] = pd.cut(df_small['mape_esmm'], bins=bins, labels=labels)

# 分布统计
dist_xgb = df_small['mape_bin_xgb'].value_counts(normalize=True).sort_index()
dist_esmm = df_small['mape_bin_esmm'].value_counts(normalize=True).sort_index()

# 组合绘图
x = range(len(labels))
plt.figure(figsize=(10, 5))
plt.bar([i - 0.2 for i in x], dist_xgb.values, width=0.4, label='XGB', color='skyblue')
plt.bar([i + 0.2 for i in x], dist_esmm.values, width=0.4, label='ESMM', color='salmon')
plt.xticks(ticks=x, labels=labels)
plt.ylabel('ratio')
plt.title('3w- MAPE')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


# In[ ]:


# xgb 模型表现 dac_nine.sab_v2_model_result
v2_y_pred
v2_pred_value


# In[ ]:


#esmm 模型表现 dac_twenty_one_dev.xy_esmm_pred
y_pred
pred_value
pred_ratio 续约概率


# In[ ]:


# Y2看 = 说这个客户能续大（>120%）、实际续大的占比，0/1 二分类问题。
续大定义：
实际回款金额 value_after0_least5 > 1.2 × 去年金额 pay_last_year → 实际续大（记为 1）


# In[14]:


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# In[38]:


# Ground truth
df['actual_big'] = (df['value_after0_least5'] > 1.2 * df['pay_last_year']).astype(int)

# XGB & ESMM 预测
df['xgb_big'] = (df['v2_pred_value'] > 1.2 * df['pay_last_year']).astype(int)
df['esmm_big'] = (df['pred_value'] > 1.2 * df['pay_last_year']).astype(int)

# -------------------------------
# 1. 分类指标报告

print("XGB 模型续大识别报告：")
print(classification_report(df['actual_big'], df['xgb_big'], digits=4))

print("ESMM 模型续大识别报告：")
print(classification_report(df['actual_big'], df['esmm_big'], digits=4))

# -------------------------------
# 2. 混淆矩阵可视化
# -------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay.from_predictions(df['actual_big'], df['xgb_big'],
                                        display_labels=["No_bigger", "Bigger"],
                                        ax=ax[0], colorbar=False)
ax[0].set_title("XGB_confusion_matrix")

ConfusionMatrixDisplay.from_predictions(df['actual_big'], df['esmm_big'],
                                        display_labels=["No_bigger", "Bigger"],
                                        ax=ax[1], colorbar=False)
ax[1].set_title("ESMM_confusion_matrix")

plt.tight_layout()
plt.show()

# -------------------------------
# 3. 按客户价值分组准确率
# -------------------------------
print(" 按 pay_last_year_bucket 分组准确率：")
print(df.groupby('pay_last_year_bucket').apply(
    lambda x: pd.Series({
        'XGB准确率': (x['xgb_big'] == x['actual_big']).mean(),
        'ESMM准确率': (x['esmm_big'] == x['actual_big']).mean()
    })
))


# 续约期 复购期 v05 假设都是不复购的状态 续约期会有差别
# 分类：
# 1.续约复购
# 2.有回款没回款
# ESMM：误报更少（FP更低）、漏报更多（FN更高）
# XGB：更“激进”地预测客户续大（FP高），但因此召回率（Recall）更高
#          |
# 

# In[39]:


# 定义 ground truth：实际是否续大
df['actual_big'] = (df['value_after0_least5'] > 1.2 * df['pay_last_year']).astype(int)

# 模型预测是否续大（>120%）
df['xgb_big'] = (df['v2_pred_value'] > 1.2 * df['pay_last_year']).astype(int)
df['esmm_big'] = (df['pred_value'] > 1.2 * df['pay_last_year']).astype(int)

# 准确率（预测是否和实际一致）
acc_xgb = (df['xgb_big'] == df['actual_big']).mean()
acc_esmm = (df['esmm_big'] == df['actual_big']).mean()

print("模型续大识别准确率：")
print("XGB 模型准确率：", round(acc_xgb, 4))
print("ESMM 模型准确率：", round(acc_esmm, 4))


# In[ ]:


# Y3看 = 说这个客户能续约（定义一个边界，如预测数字大于这个客户去年价值的80%则认为续约概率高）、实际续约的占比。


# In[40]:


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 定义 Ground Truth：实际是否续约
df['actual_renew'] = (df['value_after0_least5'] > 0).astype(int)

# 模型预测是否续约（> 80% 去年价值）
df['xgb_renew'] = (df['v2_pred_value'] > 0.8 * df['pay_last_year']).astype(int)
df['esmm_renew'] = (df['pred_value'] > 0.8 * df['pay_last_year']).astype(int)

# -------------------------------
# 分类指标
# -------------------------------
print("XGB_confusion_matrix")
print(classification_report(df['actual_renew'], df['xgb_renew'], digits=4))

print("ESMM_confusion_matrix")
print(classification_report(df['actual_renew'], df['esmm_renew'], digits=4))

# -------------------------------
# 混淆矩阵可视化
# -------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay.from_predictions(df['actual_renew'], df['xgb_renew'],
                                        display_labels=["NO_xu", "Xu"],
                                        ax=ax[0], colorbar=False)
ax[0].set_title("XGB_confusion_matrix")

ConfusionMatrixDisplay.from_predictions(df['actual_renew'], df['esmm_renew'],
                                        display_labels=["NO_xu", "Xu"],
                                        ax=ax[1], colorbar=False)
ax[1].set_title("ESMM_confusion_matrix")

plt.tight_layout()
plt.show()

# -------------------------------
# 分 bucket 准确率
# -------------------------------
print("按 pay_last_year_bucket 分组续约识别准确率：")
print(df.groupby('pay_last_year_bucket').apply(
    lambda x: pd.Series({
        'XGB准确率': (x['xgb_renew'] == x['actual_renew']).mean(),
        'ESMM准确率': (x['esmm_renew'] == x['actual_renew']).mean()
    })
))


# In[ ]:




