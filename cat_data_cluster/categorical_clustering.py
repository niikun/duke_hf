import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# サンプルデータ作成
np.random.seed(42)
n_samples = 300

# 5つの質問のサンプルデータ
data = {
    'Q1_年齢': np.random.choice(['20-29', '30-39', '40-49', '50-59', '60+'], n_samples, 
                              p=[0.2, 0.3, 0.25, 0.15, 0.1]),
    'Q2_職業': np.random.choice(['会社員', 'フリーランス', '主婦', '学生', 'その他'], n_samples,
                              p=[0.4, 0.2, 0.2, 0.1, 0.1]),
    'Q3_使用頻度': np.random.choice(['毎日', '週数回', '週1回', '月数回', 'ほとんど使わない'], n_samples,
                                p=[0.15, 0.25, 0.3, 0.2, 0.1]),
    'Q4_満足度': np.random.choice(['非常に満足', '満足', '普通', '不満', '非常に不満'], n_samples,
                               p=[0.1, 0.3, 0.4, 0.15, 0.05]),
    'Q5_継続意向': np.random.choice(['絶対続ける', '多分続ける', 'わからない', '多分やめる', '絶対やめる'], n_samples,
                                 p=[0.2, 0.35, 0.25, 0.15, 0.05])
}

df = pd.DataFrame(data)
print("サンプルデータの概要:")
print(df.head(10))
print("\n各カテゴリの分布:")
for col in df.columns:
    print(f"\n{col}:")
    print(df[col].value_counts())

# K-modesクラスタリング用にkmodes-pythonが必要
try:
    from kmodes.kmodes import KModes
    
    def perform_kmodes_clustering(df, n_clusters=3):
        """K-modesクラスタリング実行"""
        # データを数値にエンコード
        df_encoded = df.copy()
        for col in df.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
        
        # K-modesクラスタリング
        km = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=1)
        clusters = km.fit_predict(df_encoded.values)
        
        return clusters, km
    
    print("\n=== K-modesクラスタリング ===")
    clusters_kmodes, kmodes_model = perform_kmodes_clustering(df, n_clusters=3)
    df['Cluster_KModes'] = clusters_kmodes
    
    print("\nクラスタごとの分布:")
    for i in range(3):
        print(f"\n--- クラスタ {i} (n={sum(clusters_kmodes==i)}) ---")
        cluster_data = df[df['Cluster_KModes']==i]
        for col in df.columns[:-1]:
            mode_value = cluster_data[col].mode()[0]
            print(f"{col}: {mode_value} ({cluster_data[col].value_counts().iloc[0]}/{len(cluster_data)})")

except ImportError:
    print("\nkmodes-pythonパッケージが見つかりません。pip install kmodes-pythonでインストールしてください。")
    print("代わりにMCA + K-meansアプローチを使用します。")

# MCA + K-meansアプローチ
def perform_mca_kmeans_clustering(df, n_clusters=3):
    """MCA + K-meansクラスタリング"""
    # ワンホットエンコーディング
    df_encoded = pd.get_dummies(df)
    
    # 対応分析（PCAで近似）
    pca = PCA(n_components=min(10, df_encoded.shape[1]-1))
    df_pca = pca.fit_transform(df_encoded)
    
    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_pca)
    
    return clusters, kmeans, pca, df_pca

print("\n=== MCA + K-meansクラスタリング ===")
clusters_mca, kmeans_model, pca_model, df_pca = perform_mca_kmeans_clustering(df, n_clusters=3)
df['Cluster_MCA'] = clusters_mca

print(f"\n累積寄与率（最初の5成分）: {pca_model.explained_variance_ratio_[:5].cumsum()}")
print(f"シルエット係数: {silhouette_score(df_pca, clusters_mca):.3f}")

print("\nクラスタごとの分布（MCA + K-means）:")
for i in range(3):
    print(f"\n--- クラスタ {i} (n={sum(clusters_mca==i)}) ---")
    cluster_data = df[df['Cluster_MCA']==i]
    for col in ['Q1_年齢', 'Q2_職業', 'Q3_使用頻度', 'Q4_満足度', 'Q5_継続意向']:
        mode_value = cluster_data[col].mode()[0] if len(cluster_data) > 0 else 'N/A'
        count = cluster_data[col].value_counts().iloc[0] if len(cluster_data) > 0 else 0
        print(f"{col}: {mode_value} ({count}/{len(cluster_data)})")

# 可視化
plt.figure(figsize=(15, 10))

# PCAの散布図
plt.subplot(2, 3, 1)
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters_mca, cmap='viridis', alpha=0.7)
plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%})')
plt.title('MCA + K-means クラスタリング結果')
plt.colorbar(scatter)

# 各質問のクラスタ別分布
questions = ['Q1_年齢', 'Q2_職業', 'Q3_使用頻度', 'Q4_満足度', 'Q5_継続意向']
for i, question in enumerate(questions):
    plt.subplot(2, 3, i+2)
    cluster_counts = pd.crosstab(df[question], df['Cluster_MCA'], normalize='columns')
    cluster_counts.plot(kind='bar', ax=plt.gca(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(f'{question}のクラスタ別分布')
    plt.xlabel('')
    plt.ylabel('割合')
    plt.xticks(rotation=45)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# クラスタの特徴要約
print("\n=== クラスタ特徴要約 ===")
for cluster_id in range(3):
    cluster_data = df[df['Cluster_MCA'] == cluster_id]
    print(f"\nクラスタ {cluster_id} ({len(cluster_data)}人, {len(cluster_data)/len(df)*100:.1f}%):")
    
    characteristics = []
    for col in ['Q1_年齢', 'Q2_職業', 'Q3_使用頻度', 'Q4_満足度', 'Q5_継続意向']:
        mode_val = cluster_data[col].mode()[0]
        mode_pct = cluster_data[col].value_counts().iloc[0] / len(cluster_data) * 100
        if mode_pct > 40:  # 40%以上の場合のみ特徴として記載
            characteristics.append(f"{col.split('_')[1]}: {mode_val} ({mode_pct:.0f}%)")
    
    print("主な特徴:", ", ".join(characteristics))

print("\n=== 最適なクラスタ数の検討 ===")
silhouette_scores = []
k_range = range(2, 6)

for k in k_range:
    clusters_temp, _, _, df_pca_temp = perform_mca_kmeans_clustering(df, n_clusters=k)
    score = silhouette_score(df_pca_temp, clusters_temp)
    silhouette_scores.append(score)
    print(f"k={k}: シルエット係数 = {score:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_scores, 'bo-')
plt.xlabel('クラスタ数')
plt.ylabel('シルエット係数')
plt.title('クラスタ数とシルエット係数')
plt.grid(True)
plt.show()

print(f"\n最適なクラスタ数: {k_range[np.argmax(silhouette_scores)]} (シルエット係数: {max(silhouette_scores):.3f})")