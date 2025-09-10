import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')

print("First 5 rows of dataset1:\n", df1.head())
print("\nFirst 5 rows of dataset2:\n", df2.head())

df1['start_time'] = pd.to_datetime(df1['start_time'], dayfirst=True, errors='coerce')
df1['rat_period_start'] = pd.to_datetime(df1['rat_period_start'], dayfirst=True, errors='coerce')
df1['rat_period_end'] = pd.to_datetime(df1['rat_period_end'], dayfirst=True, errors='coerce')
df1['sunset_time'] = pd.to_datetime(df1['sunset_time'], dayfirst=True, errors='coerce')
df1['bat_landing_to_food'] = pd.to_numeric(df1['bat_landing_to_food'], errors='coerce')
df1['seconds_after_rat_arrival'] = pd.to_numeric(df1['seconds_after_rat_arrival'], errors='coerce')
df1['hours_after_sunset'] = pd.to_numeric(df1['hours_after_sunset'], errors='coerce')
df1 = df1.dropna(subset=['bat_landing_to_food', 'risk', 'reward', 'seconds_after_rat_arrival'])

df2['time'] = pd.to_datetime(df2['time'], dayfirst=True, errors='coerce')
df2['hours_after_sunset'] = pd.to_numeric(df2['hours_after_sunset'], errors='coerce')
df2['bat_landing_number'] = pd.to_numeric(df2['bat_landing_number'], errors='coerce')
df2['food_availability'] = pd.to_numeric(df2['food_availability'], errors='coerce')
df2['rat_minutes'] = pd.to_numeric(df2['rat_minutes'], errors='coerce')
df2['rat_arrival_number'] = pd.to_numeric(df2['rat_arrival_number'], errors='coerce')
df2 = df2.dropna(subset=['bat_landing_number', 'rat_arrival_number', 'rat_minutes'])

print("\nCleaned dataset1 shape (rows, columns):", df1.shape)
print("Cleaned dataset2 shape (rows, columns):", df2.shape)

print("\nDescriptive Statistics for dataset1:")
print(df1.describe())
print("\nProportion of Risk Behaviors (0: Avoidance, 1: Taking):")
print(df1['risk'].value_counts(normalize=True))
print("\nAverage Time to Approach Food by Risk:")
print(df1.groupby('risk')['bat_landing_to_food'].mean())

plt.figure(figsize=(8, 6))
sns.histplot(df1['bat_landing_to_food'], bins=20, kde=True)
plt.title('Distribution of Time to Approach Food (seconds)')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='risk', y='bat_landing_to_food', data=df1)
plt.title('Time to Approach Food by Risk (0: Avoidance, 1: Taking)')
plt.xlabel('Risk')
plt.ylabel('Time (seconds)')
plt.show()

df2['rat_presence'] = (df2['rat_arrival_number'] > 0).astype(int)

print("\nDescriptive Statistics for dataset2:")
print(df2.describe())
print("\nAverage Bat Landings by Rat Presence (0: No Rats, 1: Rats Present):")
print(df2.groupby('rat_presence')['bat_landing_number'].mean())

plt.figure(figsize=(8, 6))
sns.boxplot(x='rat_presence', y='bat_landing_number', data=df2)
plt.title('Bat Landings by Rat Presence (0: No Rats, 1: Rats Present)')
plt.xlabel('Rat Presence')
plt.ylabel('Number of Landings')
plt.show()

group0 = df1[df1['risk'] == 0]['bat_landing_to_food'].dropna()
group1 = df1[df1['risk'] == 1]['bat_landing_to_food'].dropna()
t_stat1, p_val1 = stats.ttest_ind(group0, group1, equal_var=False)
print(f"\nT-test for Time to Approach Food by Risk: t-stat={t_stat1:.2f}, p-value={p_val1:.4f}")
if p_val1 < 0.05:
    print("Significant difference: Bats take longer to approach food in avoidance behaviors, suggesting vigilance.")
else:
    print("No significant difference.")

no_rat = df2[df2['rat_presence'] == 0]['bat_landing_number'].dropna()
with_rat = df2[df2['rat_presence'] == 1]['bat_landing_number'].dropna()
t_stat2, p_val2 = stats.ttest_ind(no_rat, with_rat, equal_var=False)
print(f"\nT-test for Bat Landings by Rat Presence: t-stat={t_stat2:.2f}, p-value={p_val2:.4f}")
if p_val2 < 0.05:
    print("Significant difference: Fewer bat landings when rats are present, suggesting avoidance.")
else:
    print("No significant difference.")