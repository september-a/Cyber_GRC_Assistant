import pandas as pd

df_ccis = pd.DataFrame(pd.read_excel("data/NIST_800-53_Mapping.xlsx", header=1))

df_ccis = df_ccis.drop_duplicates(subset="/cci_items/cci_item/definition", keep='first')

df_controls = pd.DataFrame(pd.read_csv("data/NIST_800-53_Rev_4_Controls.csv"))

df_controls = df_controls.dropna(subset=['FAMILY'])

df_ccis = df_ccis.drop(columns=['Item', 'contributor', 'publishdate', 'creator', 'location', 'title', 'version', 'status', 'Type', 'publishdate2', 'version2'])

df_ccis['index'] = df_ccis['index'].str.replace(r'[a-z]', '', regex=True)

for index, row in df_controls.iterrows():
    filtered_df = df_ccis[df_ccis['index'].str.strip() == row['NAME']]
    filtered_df = filtered_df.drop(columns = ['Note', 'Parameter', 'index'])

    if not filtered_df.empty:
        df_string = filtered_df.to_string(index = False, header = False)
        df_string = " ".join(df_string.split())
        df_controls.loc[df_controls['NAME'] == row['NAME'], 'ccis'] = df_string

        print(filtered_df)

df_ccis.to_csv('data/NIST_800-53_Mapping.csv')

df_controls.to_csv('data/NIST_800-53_Controls.csv')

print("complete")