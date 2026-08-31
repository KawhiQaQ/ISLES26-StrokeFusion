import glob
import json
import pandas as pd

for csv_file in glob.glob(
    'path-to-isles26-trainset/*/*/*/anat/*.csv'
):
    # Read columns and rename 'SITE' to 'CENTER'
    df = pd.read_csv(csv_file, usecols=['DAYS_POST_STROKE', 'CHRONICITY', 'SITE'])
    df = df.rename(columns={'SITE': 'CENTER'})

    # Reorder columns so CENTER comes first
    df = df[['DAYS_POST_STROKE', 'CHRONICITY', 'CENTER']]

    json_path = csv_file.replace('.csv', '.json')

    if df.empty or df.shape[0] > 1:
        # Create a dictionary with reordered column names as keys and None as values
        json_output = {col: None for col in df.columns}
        with open(json_path, 'w') as json_file:
            json.dump(json_output, json_file, indent=4)
    else:
        # Convert CHRONICITY to nullable integer type
        df['CHRONICITY'] = pd.to_numeric(
            df['CHRONICITY'], errors='coerce'
        ).astype('Int64')

        # Convert the single row to a dictionary
        row_dict = df.iloc[0].to_dict()

        # Format integer for CHRONICITY, convert any NaN/NaT to None (JSON null)
        formatted_dict = {}
        for k, v in row_dict.items():
            if pd.isna(v):
                formatted_dict[k] = None
            elif k == 'CHRONICITY':
                formatted_dict[k] = int(v)
            else:
                formatted_dict[k] = v

        with open(json_path, 'w') as json_file:
            json.dump(formatted_dict, json_file, indent=4)