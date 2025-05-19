import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



# Figsize for charts:

fs_page = (12, 5)
fs_half = (8, 5)

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Read and clean data
df = pd.read_csv('data/Wine_Data.csv', sep=';')
df['Ticket'] = df['Ticket'].str.replace('€', '').str.replace(',', '.').str.strip().astype(float)

string_columns = ['Wine frequency consumption', 'Payment mode', 'Place to drink', 
                 'Additional products', 'Gender', 'Education', 'Age']
for col in string_columns:
    df[col] = df[col].str.strip()

# 1. Wine Consumption by Age (Most Important Demographic)
plt.figure(figsize=(12, 5))
age_freq = pd.crosstab(df['Age'], df['Wine frequency consumption'])
age_freq.plot(kind='barh', stacked=True)
plt.title('Wine Consumption Frequency by Age Group', fontsize=14, pad=20)
plt.xlabel('Number of Customers', fontsize=12)
plt.ylabel('Age Group', fontsize=12)
plt.legend(title='Consumption Frequency', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/wine_consumption_by_age.jpg', dpi=300, bbox_inches='tight')
plt.close()

# 2. Payment Preferences by Age
plt.figure(figsize=(12, 5))
payment_age = pd.crosstab(df['Age'], df['Payment mode'])
payment_age.plot(kind='barh', stacked=True)
plt.title('Payment Method Preferences by Age Group', fontsize=14, pad=20)
plt.xlabel('Number of Transactions', fontsize=12)
plt.ylabel('Age Group', fontsize=12)
plt.legend(title='Payment Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/payment_by_age.jpg', dpi=300, bbox_inches='tight')
plt.close()

# 3. Popular Places to Drink
plt.figure(figsize=fs_half)
place_counts = df['Place to drink'].value_counts().sort_values(ascending=False).iloc[::-1]
place_counts.plot(kind='barh')
plt.title('Most Popular Places to Drink Wine', fontsize=14, pad=20)
plt.xlabel('Number of Customers', fontsize=12)
plt.ylabel('Place', fontsize=12)
plt.tight_layout()
plt.savefig('figures/places_to_drink.jpg', dpi=300, bbox_inches='tight')
plt.close()

# 4. Average Ticket Value by Place
plt.figure(figsize=fs_half)
avg_ticket_by_place = df.groupby('Place to drink')['Ticket'].mean().sort_values(ascending=True)
avg_ticket_by_place.plot(kind='barh')
plt.title('Average Spending by Venue', fontsize=14, pad=20)
plt.xlabel('Average Ticket Value (€)', fontsize=12)
plt.ylabel('Place', fontsize=12)
plt.tight_layout()
plt.savefig('figures/avg_ticket_by_place.jpg', dpi=300, bbox_inches='tight')
plt.close()

# 5. Popular Product Pairings
plt.figure(figsize=fs_half)
product_counts = df['Additional products'].value_counts().sort_values(ascending=False).iloc[::-1]
product_counts.plot(kind='barh')
plt.title('Most Popular Additional Products', fontsize=14, pad=20)
plt.xlabel('Number of Purchases', fontsize=12)
plt.ylabel('Product', fontsize=12)
plt.tight_layout()
plt.savefig('figures/popular_products.jpg', dpi=300, bbox_inches='tight')
plt.close()

# 6. Product Preferences by Age (Heatmap)
plt.figure(figsize=fs_half)
product_age = pd.crosstab(df['Age'], df['Additional products'])
sns.heatmap(product_age, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Product Preferences by Age Group', fontsize=14, pad=20)
plt.xlabel('Additional Products', fontsize=12)
plt.ylabel('Age Group', fontsize=12)
plt.tight_layout()
plt.savefig('figures/product_age_heatmap.jpg', dpi=300, bbox_inches='tight')
plt.close()

# Histogram: Ticket Value Distribution
plt.figure(figsize=fs_half)
plt.hist(df['Ticket'], bins=20, color='grey', edgecolor='white')
plt.title('Distribution of Ticket Values')
plt.xlabel('Ticket Value (€)')
plt.ylabel('Number of Purchases')
plt.tight_layout()
plt.savefig('figures/ticket_histogram.jpg', dpi=300, bbox_inches='tight')
plt.close()

# Histogram: Frequency of Wine Consumption (Ordinal Encoding)
freq_map = {
    'Once per month': 1,
    'More than once per month': 2,
    '1 to 2 times per week': 3,
    '3 to 4 times per week': 4,
    '5 to 6 times per week': 5,
    'Once per day': 7
}
df['Freq_num'] = df['Wine frequency consumption'].map(freq_map)
plt.figure(figsize=fs_half)
plt.hist(df['Freq_num'].dropna(), bins=range(1,8), color='mediumseagreen', edgecolor='black', align='left', rwidth=0.8)
plt.title('Wine Consumption Frequency Distribution')
plt.xlabel('Frequency (higher = more often)')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.savefig('figures/frequency_histogram.jpg', dpi=300, bbox_inches='tight')
plt.close() 

# Print key statistics
print("\nKey Statistics:")
print("\n1. Average Ticket Value by Place:")
print(avg_ticket_by_place.round(2))

print("\n2. Average Ticket Value by Product:")
avg_ticket_by_product = df.groupby('Additional products')['Ticket'].agg(['mean', 'count']).sort_values('mean', ascending=False)
print(avg_ticket_by_product.round(2))

print("\n3. Payment Mode Percentages by Age Group:")
payment_percentages = payment_age.div(payment_age.sum(axis=1), axis=0) * 100
print(payment_percentages.round(1))