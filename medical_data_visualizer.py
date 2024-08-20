import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

df = pd.read_csv('medical_examination.csv', header=0)

df['overweight'] = np.where(df['weight'] / ((df['height'] / 100) ** 2) > 25, 1, 0)

df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)

df['gluc'] = np.where(df['gluc'] > 1, 1, 0)

def draw_cat_plot():

    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    order = sorted(df_cat['variable'].unique())

    g = sns.catplot(data=df_cat, kind='count', x='variable', hue='value', col='cardio', order=order)
    g.set_ylabels('total')
    fig = g.figure
    fig.savefig('catplot.png')
    return fig

def draw_heat_map():
    df_heat = df[
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] < df['height'].quantile(0.975)) &
        (df['weight'] > df['weight'].quantile(0.025)) &
        (df['weight'] < df['weight'].quantile(0.975))
    ]

    corr = df_heat.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', vmin=-0.1, vmax=0.4, cbar_kws={'shrink': 0.5}, ax=ax)

    fig.savefig('heatmap.png')
    return fig



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Read the CSV file into a DataFrame
df = pd.read_csv("6XWX_bike_rides.csv")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Filter data by year 2022
df_2022 = df[df["date"].dt.year == 2022]

# Compute and print average `group_size` in 2022
avg_group_size_2022 = df_2022["group_size"].mean()
print(f"{avg_group_size_2022}")



