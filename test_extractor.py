# test_extractor.py
from src.data_prep import extract_features_from_url

if __name__ == "__main__":
    url = "https://example.com"   # you can change this to any URL you want to test
    df = extract_features_from_url(url)
    print(df.T)        # print features vertically
    df.to_csv("test_row.csv", index=False)
    print("Saved test_row.csv")
