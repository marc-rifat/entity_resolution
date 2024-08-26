import pandas as pd
import recordlinkage

# Sample data for Dataset A
data_a = {
    'first_name': ['John', 'Jane', 'Jake', 'Emily'],
    'last_name': ['Doe', 'Doe', 'Smith', 'Jones'],
    'ssn': ['123-45-6789', '987-65-4321', '555-55-5555', '111-22-3333']
}

# Sample data for Dataset B (with some potential duplicates)
data_b = {
    'first_name': ['John', 'Jany', 'Jake', 'Emma'],
    'last_name': ['Doe', 'Doe', 'Smith', 'Jones'],
    'ssn': ['123-45-6789', '987-65-4321', '555-55-5555', '111-22-3333']
}

df_a = pd.DataFrame(data_a)
df_b = pd.DataFrame(data_b)

# Preprocess the data (e.g., cleaning the SSN by removing hyphens)
df_a['ssn_clean'] = df_a['ssn'].str.replace('-', '')
df_b['ssn_clean'] = df_b['ssn'].str.replace('-', '')

# Indexation step to reduce the number of comparisons
indexer = recordlinkage.Index()
indexer.block('ssn_clean')  # Block by SSN to ensure only records with the same SSN are compared
candidate_links = indexer.index(df_a, df_b)

# Comparison step to compare the records
compare = recordlinkage.Compare()

# Compare first name and last name using string similarity (e.g., Levenshtein distance)
compare.string('first_name', 'first_name', method='levenshtein', threshold=0.85, label='first_name')
compare.string('last_name', 'last_name', method='levenshtein', threshold=0.85, label='last_name')

# Exact match comparison for SSN
compare.exact('ssn_clean', 'ssn_clean', label='ssn')

# Get the comparison features
features = compare.compute(candidate_links, df_a, df_b)

# Classify the potential matches using a simple rule-based approach
matches = features[(features['first_name'] > 0.85) & (features['last_name'] > 0.85) & (features['ssn'] == 1)]

# Print the matching names and SSNs
print("Matching Records:")

for index_a, index_b in matches.index:
    record_a = df_a.loc[index_a]
    record_b = df_b.loc[index_b]
    print(f"Match Found: {record_a['first_name']} {record_a['last_name']} ({record_a['ssn']}) "
          f"<-> {record_b['first_name']} {record_b['last_name']} ({record_b['ssn']})")
