import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

baslik_font = {'family': 'arial', 'color': 'darkred','weight': 'bold','size': 13 }
eksen_font  = {'family': 'arial', 'color': 'darkblue','weight': 'bold','size': 10 }


#proje2 = Abd'deki ev fiyatları.
# house_df = pd.read_csv("final_dataa.csv")
# # print(house_df.info())
# house_df['zindexvalue'] = house_df['zindexvalue'].str.replace(',', '')
# house_df["zindexvalue"]=house_df["zindexvalue"].astype(np.int64)
#
#
# # K Neighbors Regresyon
# print("Ağırlıksız regresyon")
# for i in range(1,15):
#     knn = KNeighborsRegressor(n_neighbors=i)
#     X = house_df[["bathrooms", "bedrooms","finishedsqft","totalrooms","yearbuilt","zestimate","zindexvalue"]]
#     y = house_df.lastsoldprice
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 40)
#     knn.fit(X_train, y_train)
#
#     print("Neighbor sayısı:",i ,"           score:",knn.score(X_test, y_test))
#     cvscores_3 = cross_val_score(knn, X_test, y_test, cv=3)
#     print('cv= 3, Ort çarpraz doğrulama skoru : ', np.mean(cvscores_3))
#     cvscores_10 = cross_val_score(knn, X_test, y_test, cv=10)
#     print('cv=10, Ort çarpraz doğrulama skoru : ', np.mean(cvscores_10))
#     print()
#
#
# knn = KNeighborsRegressor(n_neighbors=5)
# # , "bedrooms","finishedsqft","totalrooms","yearbuilt","zestimate","zindexvalue"
# X = house_df[["bathrooms"]]
# y = house_df.lastsoldprice
# knn.fit(X, y)
# T = np.arange(0, 16, 0.1)[:, np.newaxis]
# y_tahmin = knn.predict(T)
# plt.scatter(X, y, c='darkblue', label='Veriler')
# plt.plot(T, y_tahmin, c='darkgreen', label='Tahminler')
# plt.legend()
# plt.title('Komşu Sayısı=5, Ağırlıksız', fontdict = baslik_font)
# plt.xlabel('Banyo Sayısı', fontdict = eksen_font)
# plt.ylabel('Fiyat', fontdict = eksen_font)
# plt.show()
# ##################################################################################################################
#
# print("\n\nAğırlıklı regresyon")
# for i in range(1,15):
#     knn_agirlik = KNeighborsRegressor(n_neighbors=i, weights='distance')
#     X = house_df[["bathrooms", "bedrooms","finishedsqft","totalrooms","yearbuilt","zestimate","zindexvalue"]]
#     y = house_df.lastsoldprice
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     knn_agirlik.fit(X, y)
#
#     print("Neighbor sayısı:", i, " score:", knn_agirlik.score(X_test, y_test))
#
#
# knn = KNeighborsRegressor(n_neighbors=5, weights="distance")
# # , "bedrooms","finishedsqft","totalrooms","yearbuilt","zestimate","zindexvalue"
# X = house_df[["bathrooms"]]
# y = house_df.lastsoldprice
# knn.fit(X, y)
# T = np.arange(0, 16, 0.1)[:, np.newaxis]
# y_tahmin = knn.predict(T)
# plt.scatter(X, y, c='darkblue', label='Veriler')
# plt.plot(T, y_tahmin, c='darkgreen', label='Tahminler')
# plt.legend()
# plt.title('Komşu Sayısı=5, Ağırlıksız', fontdict = baslik_font)
# plt.xlabel('Banyo Sayısı', fontdict = eksen_font)
# plt.ylabel('Fiyat', fontdict = eksen_font)
# plt.show()

##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################


#Proje3 fraud credit card alışverişleri

#Sınıflandırma not-weighted
#Imbalanced veriyi balanced yaptım.

# proje3_df = pd.read_csv("creditcard.csv")
#
# normal_alısveris = proje3_df[proje3_df.Class == 0]
# sahte_alısveris = proje3_df[proje3_df.Class == 1]
#
# normal_alısveris_azaltılmış = resample(normal_alısveris,
#                                      replace = True,
#                                      n_samples = len(sahte_alısveris),
#                                      random_state = 111)
#
# azaltılmış_df = pd.concat([sahte_alısveris, normal_alısveris_azaltılmış])
# print(azaltılmış_df.Class.value_counts())
#
#
# for i in range(1,15):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     X = azaltılmış_df.drop('Class', axis=1)
#     y = azaltılmış_df['Class']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     knn.fit(X_train, y_train)
#
#     cvscores_3 = cross_val_score(knn, X_test, y_test, cv=3)
#     print('cv= 3, Ort çarpraz doğrulama skoru : ', np.mean(cvscores_3))
#
#     cvscores_10 = cross_val_score(knn, X_test, y_test, cv=10)
#     print('cv= 10, Ort çarpraz doğrulama skoru : ', np.mean(cvscores_10))
#
#     print("Neighbor sayısı:", i, " score:", knn.score(X_test, y_test))
#     print()


# #################################################################################################################

# for i in range(1,15):
#     knn_agirlik = KNeighborsClassifier(n_neighbors=i, weights='distance')
#     X = azaltılmış_df.drop('Class', axis=1)
#     y = azaltılmış_df['Class']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     knn_agirlik.fit(X_train, y_train)
#
#     cvscores_3 = cross_val_score(knn_agirlik, X_test, y_test, cv=3)
#     print('cv= 3, Ort çarpraz doğrulama skoru : ', np.mean(cvscores_3))
#
#     cvscores_10 = cross_val_score(knn_agirlik, X_test, y_test, cv=10)
#     print('cv= 10, Ort çarpraz doğrulama skoru : ', np.mean(cvscores_10))
#
#     print("Neighbor sayısı:", i, " score:", knn_agirlik.score(X_test, y_test))
#     print()
