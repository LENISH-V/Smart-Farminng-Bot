import requests
import os
import xml.etree.ElementTree as ET
import time
from PyPDF2 import PdfReader

def fetch_paper_urls_from_arxiv(keywords, paper_limit=1000):
    """
    Fetch URLs of open-access papers from arXiv in .pdf format.
    """
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{keywords}",
        "start": 0,
        "max_results": paper_limit
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.text

        # Parse the XML response
        root = ET.fromstring(data)
        paper_urls = []

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            link = entry.find("{http://www.w3.org/2005/Atom}id")
            if link is not None:
                pdf_url = link.text.strip().replace("http://arxiv.org/abs/", "http://arxiv.org/pdf/") + ".pdf"
                paper_urls.append(pdf_url)

        return paper_urls
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from arXiv: {e}")
        return []

def convert_pdf_to_text(pdf_path, txt_path):
    """
    Convert a PDF file to a text file.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            if page is not None:
                text += page.extract_text() or ""
        with open(txt_path, "w", encoding="utf-8") as file:
            file.write(text.strip())
    except Exception as e:
        print(f"Error converting {pdf_path} to text: {e}")

def download_papers(paper_urls, download_dir="downloaded_papers"):
    """
    Download papers in .pdf format from the provided URLs and convert them to .txt files.
    Only download papers that are less than 4MB in size.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for i, url in enumerate(paper_urls):
        print(f"Processing paper {i + 1}: {url}")  # Debugging log
        try:
            time.sleep(3)  # Pause between requests to avoid being blocked

            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Check content length to ensure the file is less than 4MB
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    content_length = int(content_length)
                    if content_length > 4 * 1024 * 1024:  # Corrected to 4MB
                        print(f"Skipping {url}: File size exceeds 4MB.")
                        continue
                except ValueError:
                    print(f"Skipping {url}: Invalid Content-Length header.")
                    continue

            pdf_path = os.path.join(download_dir, f"paper_{i + 1}.pdf")
            with open(pdf_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Downloaded: {pdf_path}")

            # Convert the downloaded PDF to a text file
            txt_path = os.path.join(download_dir, f"paper_{i + 1}.txt")
            convert_pdf_to_text(pdf_path, txt_path)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading paper {i + 1} ({url}): {e}")
        except Exception as e:
            print(f"Unexpected error processing {url}: {e}")

# Example usage
if __name__ == "__main__":
    keywords = "agriculture OR pest OR insect OR soil OR weather OR climate OR temperature OR fertilizer OR manure OR compost"
    paper_limit = 1000

    paper_urls = fetch_paper_urls_from_arxiv(keywords, paper_limit=paper_limit)
    if paper_urls:
        print(f"Total Papers Found: {len(paper_urls)}")
        download_papers(paper_urls)
    else:
        print("No papers found or failed to fetch URLs.")
        