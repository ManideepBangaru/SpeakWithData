import pandas as pd

def getCSV(file, cached_df, cached_file_path):
    try:
        # Check if the file path has changed
        if cached_file_path != file:
            # Load the uploaded file into a DataFrame
            try:
                cached_df = pd.read_csv(file)
                cached_file_path = file
                print("****************** dataframe loaded successfully ******************\n")
                return cached_df, cached_file_path
            except Exception as e:
                return f"Error loading the dataset: {e}", None
        else:
            print("****************** Using cached dataframe ******************\n")
            return cached_df, cached_file_path
    except Exception as e:
        print(e)
        