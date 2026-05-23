import requests
import os

# List of book IDs (find more at gutenberg.org, e.g., Mother Goose is 17208)
book_ids = [
    10607,
    39784,
    26197
]  # Add more IDs here for vastness

os.makedirs("dataset", exist_ok=True)  # Create a folder

for book_id in book_ids:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"dataset/book_{book_id}.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Downloaded book {book_id}")
    else:
        print(f"Failed to download {book_id}")
