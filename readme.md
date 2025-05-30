# Automated Google Scholar PDF Scraper

## Overview

This Python project automates the search, download, and metadata extraction of academic PDF documents from Google Scholar based on complex Boolean keyword queries. It is designed for researchers and analysts who need to gather large sets of scholarly articles efficiently and reliably.

## Key Features

1. **Advanced Boolean Query Support**  
   - Parses AND/OR logic in your query string to generate all term combinations and maximize result coverage.

2. **Targeted PDF Search**  
   - Appends `filetype:pdf` to each query to ensure only PDF documents are returned.

3. **Anti-Scraping Measures**  
   - **Rotating User-Agents**: Randomizes headers on every request to mimic different browsers.  
   - **Randomized Delays**: Inserts human-like pauses (5–10 seconds) between searches and downloads.  
   - **Adaptive Back-off**: Detects HTTP 429 and CAPTCHAs and automatically pauses for 5 minutes.

4. **Proxy Support**  
   - Optional integration with HTTP/S proxies for IP rotation and enhanced anonymity.

5. **Robust Download Validation**  
   - Verifies PDF signature (`%PDF-`) and HTTP status code before saving.  
   - Sanitizes filenames to ensure Windows compatibility.

6. **Comprehensive Metadata Extraction**  
   - Uses PyMuPDF to extract text and PDF metadata (title, author, creation date).  
   - Heuristic and NLP-based approaches to guess author, publication date, theme, and module.  
   - Extracts abstracts and keyword sections from the first two pages.

7. **Link Extraction**  
   - Collects URLs from link annotations and text, filtered for academic domains (`.edu`, `.ac`, `.org`).

8. **Caching and Duplication Avoidance**  
   - Loads existing Excel records to skip already-downloaded PDF URLs.

9. **Formatted Excel Reporting**  
   - Appends new results to an existing Excel file, preserves prior data.  
   - Automatically adjusts column widths, freezes headers, and applies filters for easy review.

10. **Summary Reporting**  
    - Prints total downloads, failures, and a breakdown of detected document themes.

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/scholar-pdf-scraper.git
   cd scholar-pdf-scraper
