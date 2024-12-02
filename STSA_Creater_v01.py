"""
STテーブルとガンマターゲットを元にSTSAテーブルを作成する

"""
import pandas as pd
import Gamma_Tgt_Creater
import numpy as np
import matplotlib.pyplot as plt
import os

# STテーブルを読み込む
#df = pd.read_csv('./tif/deltaE_Y_gijyututen241105_19_256_st_table.tif')
#print(df)
folder_path = r'C:\Users\aaa038498\PycharmProjects\ImageProcessing\GammaCorrection/csv'
csv_filename = 'deltaE_M_19_for_RD_fitting_256_st_table.csv' # ★★★
df = pd.read_csv(os.path.join(folder_path,csv_filename))


# ガンマターゲットを読み込む
df_linear =Gamma_Tgt_Creater.create_linier_tgt()
print(df_linear)

# fittingテーブル(7次関数の係数を保存するテーブル)(21×8)を作成する
df_fitting = pd.DataFrame(columns=[])

# fittingを行い、7次関数の係数をデータフレームに保存する
sp = df.T.index
for t in sp:
    f = np.polyfit(df[str(t)],df.index,7)
    df_fitting[str(t)] = f
print(df_fitting)

# 関数(列番号とdeltaE*を入力するとDataが出てくる関数)を作成する
def func(col,deltaE):
    data = df_fitting[str(col)][0]*deltaE**7+df_fitting[str(col)][1]*deltaE**6+df_fitting[str(col)][2]*deltaE**5+df_fitting[str(col)][3]*deltaE**4+df_fitting[str(col)][4]*deltaE**3+df_fitting[str(col)][5]*deltaE**2+df_fitting[str(col)][6]*deltaE+df_fitting[str(col)][7]
    data_int = int(data)
    return data_int

# 空のSTSAテーブルを作成する（21×256）

data = df.index
df_stsa = pd.DataFrame(index=data,columns=[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10])
df_stsa.index = df.index

for col in [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]:
    data_list = []
    for i in range(256):
        deltaE = df_linear['0'][i]
        data_list.append(func(col, deltaE))
    df_stsa[col] = data_list

# data0と255の処理
df_stsa.iloc[0] = 0
df_stsa.iloc[255] = 255

# 255以上の値を255にする
df_stsa = df_stsa.clip(upper=255)

print(df_stsa)

# STSAの表示
fig, ax = plt.subplots(nrows=1, ncols=1)

for l in [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]:
    ax.plot(data,df_stsa[l])
    ax.set_xlabel('data')
    ax.set_ylabel('corrected_data')

#plt.show()

# 補正後の長手のガンマ（L＊）データの表示

fig, ax = plt.subplots(nrows=1, ncols=1)
for j in range(256):
    ax.plot(df.T.index[1:],df.T[int(j)][1:])
    ax.set_xlabel('position')
    ax.set_ylabel('L*')
plt.show()

#df_stsa.to_csv('./tif/stsa_demo241010_table.tif' ,header=False, index=False)
csv_filename_without_extension = os.path.splitext(csv_filename)[0]
output_csv_filename = csv_filename_without_extension +'_stsa_table.csv'

# csv出力
df_stsa.to_csv(os.path.join(folder_path,output_csv_filename), index=False)