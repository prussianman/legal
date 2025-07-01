def extract_timestamp(x):
    if isinstance(x, list) and len(x) > 0:
        return x[0]  # Take the first item from the list
    return x # Return the item as-is if it's not a list

final_df['decision_date'] = final_df['decision_date'].apply(extract_timestamp)


# Step 2: Now, ensure everything is a proper datetime object.
# This cleans up any remaining strings or other types, turning errors into NaT (Not a Time).
final_df['decision_date'] = pd.to_datetime(final_df['decision_date'], errors='coerce')


# Step 3: With a clean column, safely convert to the ISO string format.
# This handles the NaT null values correctly.
final_df['decision_date'] = final_df['decision_date'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
