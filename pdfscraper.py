import os
import time
import random
import requests
import fitz  # PyMuPDF
import pandas as pd
from scholarly import scholarly
from fake_useragent import UserAgent
import re

# Initialize user agent for requests
ua = UserAgent()

HEADERS = {'User-Agent': ua.random}

def sanitize_filename(filename):
    # Remove invalid characters for Windows filenames
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def get_new_headers():
    """Return a new randomized User-Agent header."""
    return {'User-Agent': UserAgent().random}

def search_google_scholar(keyword, max_results=20, delay=5, proxies=None):
    """Search Google Scholar for articles matching the keyword, targeting only PDFs. Rotates user agent and supports proxies and adaptive back-off."""
    results = []
    # Refine the search query to include filetype:pdf
    refined_keyword = f"{keyword} filetype:pdf"
    search_query = scholarly.search_pubs(refined_keyword)

    for i in range(max_results):
        try:
            # Rotate user agent every search
            global HEADERS
            HEADERS = get_new_headers()
            article = next(search_query)
            result = {
                'title': article.get('bib', {}).get('title'),
                'author': article.get('bib', {}).get('author'),
                'year': article.get('bib', {}).get('pub_year'),
                'abstract': article.get('bib', {}).get('abstract'),
                'link': article.get('pub_url'),
                'pdf_url': article.get('eprint_url')
            }
            results.append(result)
            print(f"[{i+1}] Fetched: {result['title']}")
            # Randomize pause between each search
            time.sleep(random.uniform(5, 10))
        except StopIteration:
            break
        except Exception as e:
            # Adaptive back-off for HTTP 429 or CAPTCHA
            err_str = str(e).lower()
            if '429' in err_str or 'captcha' in err_str:
                print('⚠️ Detected rate limit or CAPTCHA. Backing off for 5 minutes...')
                time.sleep(300)
            else:
                print(f" Error fetching article {i+1}: {e}")
                time.sleep(delay)

    return results

# Simplified download_pdf: only checks for PDF signature, assumes URL is for a PDF (since search is refined)
def download_pdf(url, title, folder="pdfs", proxies=None):
    """Download a PDF from a URL if it is a valid PDF. Rotates user agent and supports proxies and adaptive back-off."""
    try:
        os.makedirs(folder, exist_ok=True)
        safe_title = sanitize_filename(title[:50].replace(' ', ''))
        filename = f"{folder}/{safe_title}.pdf"
        headers = get_new_headers()
        response = requests.get(url, headers=headers, timeout=15, proxies=proxies)
        # Only check for PDF signature and status code
        if response.status_code == 200 and response.content[:5] == b'%PDF-':
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"\U0001F4E5 PDF saved: {filename}")
            return filename
        elif response.status_code == 429 or b'captcha' in response.content.lower():
            print('⚠️ Detected rate limit or CAPTCHA while downloading PDF. Backing off for 5 minutes...')
            time.sleep(300)
        else:
            print(f"❌ Not a valid PDF (status: {response.status_code}, signature: {response.content[:5]}): {url}")
    except Exception as e:
        print(f"❌ Failed to download PDF: {e}")
    return None

def extract_links(text, doc=None):
    """
    Extracts URLs from the PDF using both link annotations and text, post-filtered for academic domains and page range.
    """
    links = set()
    # Use link annotations if doc is provided
    if doc is not None:
        for page in doc:
            for l in page.get_links():
                uri = l.get('uri')
                if uri:
                    links.add(uri)
    # Fallback: regex from first/last 2 pages of text
    pages = text.split('\f')
    sample_text = '\n'.join(pages[:2] + pages[-2:]) if len(pages) > 2 else text
    links.update(re.findall(r'https?://\S+', sample_text))
    # Post-filter for academic domains
    academic_domains = ('.edu', '.ac', '.org')
    filtered = [url for url in links if any(url.lower().endswith(d) or d in url.lower() for d in academic_domains)]
    return filtered or list(links)

def guess_author(text, doc=None):
    """
    Try PDF metadata, then headings, then NER, then fallback to 'by' pattern.
    """
    # 1. PDF metadata
    if doc is not None:
        meta = doc.metadata
        if meta and meta.get('author') and meta['author'].strip():
            return meta['author'].strip()
    # 2. Headings
    for line in text.split('\n')[:20]:
        if re.match(r'author[s]?:', line, re.I):
            return line.split(':', 1)[-1].strip()
    # 3. NER (spaCy, if available)
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        doc_nlp = nlp('\n'.join(text.split('\n')[:20]))
        people = [ent.text for ent in doc_nlp.ents if ent.label_ == 'PERSON']
        if people:
            return ', '.join(people)
    except Exception:
        pass
    # 4. Fallback: 'by' pattern
    for line in text.split('\n')[:10]:
        if 'by' in line.lower():
            return line.strip()
    return '?unknown'

def guess_date(text, doc=None):
    """
    Try PDF metadata, then full date regex, then fallback to year regex (first 2 pages).
    """
    # 1. PDF metadata
    if doc is not None:
        meta = doc.metadata
        for k in ('creationDate', 'modDate'):
            if meta and meta.get(k):
                m = re.search(r'(19|20)\d{2}', meta[k])
                if m:
                    return m.group(0)
    # 2. Full date (e.g. March 2022)
    for line in text.split('\n')[:30]:
        m = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(19|20)\d{2}', line)
        if m:
            return m.group(0)
    # 3. Year regex (first 2 pages)
    pages = text.split('\f')
    sample_text = '\n'.join(pages[:2]) if len(pages) > 2 else text
    m = re.search(r'(19|20)\d{2}', sample_text)
    return m.group(0) if m else '?unknown'

def guess_theme(text, theme_dict=None):
    """
    Use a larger theme dictionary or TF-IDF/embedding if available, else fallback to keyword search.
    """
    # 1. Use provided theme_dict (keyword: label)
    if theme_dict:
        for label, keywords in theme_dict.items():
            for kw in keywords:
                if kw.lower() in text.lower():
                    return label
    # 2. Fallback: hardcoded
    for keyword in ["AI", "Machine Learning", "Blockchain", "Security", "Education"]:
        if keyword.lower() in text.lower():
            return keyword
    return '?unknown'

def guess_module(text):
    """
    Look for headings, section/chapter regex, or known module names.
    """
    # 1. Section/Chapter regex
    for line in text.split('\n'):
        if re.match(r'(section|chapter|unit)\s*\d+', line, re.I):
            return line.strip()
    # 2. Headings
    for keyword in ["Module", "Course", "Unit", "Chapter"]:
        for line in text.split('\n'):
            if keyword.lower() in line.lower():
                return line.strip()
    return '?unknown'

def extract_pdf_info(filepath):
    """Extract information from a PDF file using PyMuPDF and improved heuristics, including more metadata."""
    try:
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        meta = doc.metadata or {}
        # Try to extract abstract and keywords from the first 2 pages
        abstract, keywords = '', ''
        first_pages = text.split('\f')[:2]
        first_text = '\n'.join(first_pages)
        # Abstract
        abs_match = re.search(r'(?i)abstract[:\s\n]+(.{30,1000})', first_text)
        if abs_match:
            abstract = abs_match.group(1).split('\n')[0].strip()
        # Keywords
        kw_match = re.search(r'(?i)keywords?[:\s\n]+([\w,;\- ]{3,200})', first_text)
        if kw_match:
            keywords = kw_match.group(1).strip()
        info = {
            "pdf_path": filepath,
            "found_links": extract_links(text, doc),
            "guessed_author": guess_author(text, doc),
            "guessed_date": guess_date(text, doc),
            "guessed_theme": guess_theme(text),
            "guessed_module": guess_module(text),
            # Additional metadata
            "meta_title": meta.get('title', ''),
            "meta_author": meta.get('author', ''),
            "meta_subject": meta.get('subject', ''),
            "meta_creation": meta.get('creationDate', ''),
            "meta_mod": meta.get('modDate', ''),
            "meta_keywords": meta.get('keywords', ''),
            "extracted_abstract": abstract,
            "extracted_keywords": keywords
        }
        return info
    except Exception as e:
        print(f"❌ Error reading PDF {filepath}: {e}")
        return None

def parse_and_or_query(query):
    """
    Parse a query string with AND/OR logic and return all combinations for search.
    Example: 'A AND (B OR C) AND (D OR E OR F)' ->
    [['A'], ['B', 'C'], ['D', 'E', 'F']] ->
    ['A B D', 'A B E', 'A B F', 'A C D', 'A C E', 'A C F']
    """
    import re
    # Split by AND (case-insensitive)
    and_groups = re.split(r'\s+AND\s+', query, flags=re.I)
    or_terms = []
    for group in and_groups:
        # Remove parentheses
        group = group.strip()
        if group.startswith('(') and group.endswith(')'):
            group = group[1:-1]
        # Split by OR (case-insensitive)
        terms = [t.strip(' ()"') for t in re.split(r'\s+OR\s+', group, flags=re.I)]
        or_terms.append([t for t in terms if t])
    # Cartesian product
    from itertools import product
    combos = list(product(*or_terms))
    # Join each combo into a query string (quoted for multi-word terms)
    queries = [' '.join(f'"{term}"' if ' ' in term else term for term in combo if term) for combo in combos]
    return queries

def main(query_string, max_results=20, proxies=None):
    """Main function to search, download, extract, and save results for all combinations from the parsed query string."""
    all_extracted_data = []
    excel_path = "C:\\Users\\Amine\\desktop\\pdfsss\\extracted_results.xlsx"
    # Load cached links if file exists
    cached_links = set()
    if os.path.exists(excel_path):
        try:
            existing_df = pd.read_excel(excel_path)
            if 'PDF URL' in existing_df.columns:
                cached_links = set(existing_df['PDF URL'].dropna().astype(str))
            elif 'pdf_url' in existing_df.columns:
                cached_links = set(existing_df['pdf_url'].dropna().astype(str))
        except Exception as e:
            print(f"Warning: Could not read existing Excel for cache: {e}")
    download_count = 0
    fail_count = 0
    topic_counter = {}
    queries = parse_and_or_query(query_string)
    for keyword in queries:
        print(f"\n🔍 Searching for: {keyword}")
        results = search_google_scholar(keyword, max_results, proxies=proxies)
        extracted_data = []
        for article in results:
            pdf_url = article.get("pdf_url")
            if not pdf_url:
                print(f"Skipped: No PDF URL for article '{article.get('title')}'")
                fail_count += 1
                continue
            if pdf_url in cached_links:
                print(f"⏩ Skipped (already downloaded): {pdf_url}")
                continue
            title = article.get("title", "untitled")
            folder_path = "C:\\Users\\Amine\\desktop\\pdfsss"
            filepath = download_pdf(pdf_url, title, folder=folder_path, proxies=proxies)
            if not filepath:
                print(f"Skipped: Could not download or validate PDF for '{title}'")
                fail_count += 1
                continue
            pdf_info = extract_pdf_info(filepath)
            if not pdf_info:
                print(f"Skipped: Could not extract info from PDF '{title}'")
                fail_count += 1
                continue
            # Count topics
            topic = pdf_info.get('guessed_theme', '?unknown')
            topic_counter[topic] = topic_counter.get(topic, 0) + 1
            article.update(pdf_info)
            extracted_data.append(article)
            download_count += 1
        print(f"Total valid PDFs extracted for '{keyword}': {len(extracted_data)}")
        all_extracted_data.extend(extracted_data)
    if not all_extracted_data:
        print("No valid PDFs extracted for any keyword.")
        return
    df = pd.DataFrame(all_extracted_data)
    # Reorder and rename columns for presentation
    column_order = [
        'title', 'author', 'year', 'guessed_author', 'guessed_date', 'guessed_theme', 'guessed_module',
        'abstract', 'link', 'pdf_url', 'pdf_path', 'found_links'
    ]
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    df = df.rename(columns={
        'title': 'Title',
        'author': 'Authors',
        'year': 'Year',
        'guessed_author': 'Extracted Author',
        'guessed_date': 'Extracted Year',
        'guessed_theme': 'Theme',
        'guessed_module': 'Module',
        'abstract': 'Abstract',
        'link': 'Scholar Link',
        'pdf_url': 'PDF URL',
        'pdf_path': 'PDF Path',
        'found_links': 'Links in PDF'
    })
    if 'Links in PDF' in df.columns:
        df['Links in PDF'] = df['Links in PDF'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    if os.path.exists(excel_path):
        existing_df = pd.read_excel(excel_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
        workbook = writer.book
        worksheet = writer.sheets['Results']
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, min(max_len, 50))
        worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
        worksheet.freeze_panes(1, 0)
    print("✅ Extraction complete. Results saved to 'extracted_results.xlsx'.")
    print("\n===== Extraction Summary =====")
    print(f"Total PDFs downloaded: {download_count}")
    print(f"Total failed/skipped: {fail_count}")
    print("Topics detected across PDFs:")
    for topic, count in topic_counter.items():
        print(f"  {topic}: {count}")
    print("============================\n")

if __name__ == "__main__":
    # Example: pass your query string with AND/OR logic
    query_string = (
        '("Tunisie" OR "Tunis" OR "Grand Tunis" OR "tunisien" OR "Maghreb") '
        'AND ("habitat anarchique" OR "habitat spontané" OR "habitat non réglementaire" '
        'OR "habitat informel" OR "gourbiville" OR "bidonville" OR "habitat précaire" '
        'OR "habitat insalubre" OR "lotissement clandestin" OR "oukala" '
        'OR "Habitat spontané péri-urbain" OR "HSPU" OR "gourbi" OR "quartier informel" '
        'OR "quartier non réglementaire" OR "quartier anarchique")'
    )
    proxies = None  # Set your proxies here if needed
    main(query_string=query_string, max_results=20, proxies=proxies)