import pandas as pd

# 读取四个csv文件
csv3 = pd.read_csv('/data/yanglongjiang/project/FastMER/predictions_tack1_wei_qq3.csv')
csv4 = pd.read_csv('/data/yanglongjiang/project/FastMER/predictions_tack1_wei_qq4.csv')
csv5 = pd.read_csv('/data/yanglongjiang/project/FastMER/predictions_tack1_wei_qq5.csv')
csv6 = pd.read_csv('/data/yanglongjiang/project/FastMER/predictions_tack1_wei_qq6.csv')

# 合并，保持id顺序与csv3一致
submission = pd.DataFrame()
submission['id'] = csv3['id']
submission['Honesty-Humility'] = csv3.iloc[:, 1]
submission['Extraversion'] = csv4.set_index('id').loc[submission['id']].iloc[:, 0].values
submission['Agreeableness'] = csv5.set_index('id').loc[submission['id']].iloc[:, 0].values
submission['Conscientiousness'] = csv6.set_index('id').loc[submission['id']].iloc[:, 0].values

submission.to_csv('../result/submission_track1_wei.csv', index=False)
print('合并完成，已保存为 submission_track1_2.csv')