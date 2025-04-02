import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def extract_text_from_txt(txt_path):
    """
    Extract text from a .txt file.
    """
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading text from {txt_path}: {e}")
        return ""

def cluster_papers_by_keywords(download_dir="downloaded_papers", output_dir="keyword_clusters"):
    """
    Cluster papers into four categories based on the keywords: fertilizer, pest, soil, and weather.
    """
    keywords = ["fertilizer", "pest", "soil", "weather"]
    file_texts = []
    file_names = []

    # Extract text from all .txt files
    for file_name in os.listdir(download_dir):
        if file_name.endswith(".txt"):
            txt_path = os.path.join(download_dir, file_name)
            text = extract_text_from_txt(txt_path)
            if text:
                file_texts.append(text)
                file_names.append(file_name)

    # Perform clustering using K-Means
    if file_texts:
        vectorizer = TfidfVectorizer(stop_words="english", vocabulary=keywords)
        X = vectorizer.fit_transform(file_texts)
        kmeans = KMeans(n_clusters=4, random_state=42)
        labels = kmeans.fit_predict(X)

        # Create cluster directories
        cluster_dirs = [os.path.join(output_dir, keyword) for keyword in keywords]
        for cluster_dir in cluster_dirs:
            os.makedirs(cluster_dir, exist_ok=True)

        # Save files into their respective cluster folders
        for file_name, label in zip(file_names, labels):
            txt_path = os.path.join(download_dir, file_name)
            cluster_dir = cluster_dirs[label]
            os.rename(txt_path, os.path.join(cluster_dir, file_name))
            print(f"File {file_name} moved to {cluster_dir}")
    else:
        print("No valid text extracted from .txt files.")

# Example usage
if __name__ == "__main__":
    download_dir = "downloaded_papers"
    output_dir = "keyword_clusters"
    cluster_papers_by_keywords(download_dir, output_dir)
