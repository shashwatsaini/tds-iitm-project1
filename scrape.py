import os
import time
import pickle
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup  
from rapidfuzz import fuzz

TDS_COURSE_PAGE_SCRAPED_FILE = 'tds_course_page_scraped.pkl'
TDS_DISCOURSE_PAGE_SCRAPED_FILE = 'tds_discourse_page_scraped.pkl'

DISCOURSE_EMAIL = os.environ['DISCOURSE_EMAIL']
DISCOURSE_PASSWORD = os.environ['DISCOURSE_PASSWORD']

def query_tds_course_page(query, threshold=60):
    """
    Returns the URL of the page with the highest similarity score above threshold.
    Uses fuzzy matching to find the most probable page for the query.
    """
    if os.path.exists(TDS_COURSE_PAGE_SCRAPED_FILE):
        with open(TDS_COURSE_PAGE_SCRAPED_FILE, 'rb') as f:
            visited_pages = pickle.load(f)
        
        normalized_query = query.lower().strip()
        best_score = 0
        best_url = None

        for url, content in visited_pages.items():
            normalized_content = content.lower().strip()
            # Compute similarity ratio
            score = fuzz.partial_ratio(normalized_query, normalized_content)
            if score > best_score:
                best_score = score
                best_url = url
        
        if best_score >= threshold:
            return best_url
        else:
            return None
    else:
        return None
    
def query_tds_discourse_page(query, threshold=60):
    """
    Returns the URL of the page with the highest similarity score above threshold.
    Uses fuzzy matching to find the most probable page for the query.
    """
    if os.path.exists(TDS_DISCOURSE_PAGE_SCRAPED_FILE):
        with open(TDS_DISCOURSE_PAGE_SCRAPED_FILE, 'rb') as f:
            visited_pages = pickle.load(f)
        
        normalized_query = query.lower().strip()
        best_score = 0
        best_url = None

        for url, content in visited_pages.items():
            normalized_content = content.lower().strip()
            # Compute similarity ratio
            score = fuzz.partial_ratio(normalized_query, normalized_content)
            if score > best_score:
                best_score = score
                best_url = url
        
        if best_score >= threshold:
            return best_url
        else:
            return None
    else:
        return None

def scrape_tds_course_page():
    """
    Scrape the TDS course page to collect all unique links and their rendered HTML content.
    If the content has already been scraped, load it from a pickle file.
    
    Returns:
        Dict[str, str]: A dictionary mapping each visited URL to its rendered HTML content.
    """

    # Check if the cache file exists
    if os.path.exists(TDS_COURSE_PAGE_SCRAPED_FILE):
        with open(TDS_COURSE_PAGE_SCRAPED_FILE, 'rb') as f:
            visited_pages = pickle.load(f)
        print(f"Loaded {len(visited_pages)} pages from cached HTML.")
        return visited_pages
    
    # Setup headless browser
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=options)

    # Base site
    base_url = "https://tds.s-anand.net"

    # Load main page
    driver.get(base_url)
    time.sleep(2)  # wait for JS to load

    # Collect links like "#/page1"
    links = set()
    anchors = driver.find_elements(By.TAG_NAME, "a")
    for a in anchors:
        href = a.get_attribute("href")
        if href and "#/" in href:
            links.add(href)

    # Visit each link
    visited_pages = {}
    for link in links:
        print(f"Visiting: {link}")
        driver.get(link)
        time.sleep(2)  # wait for JS content to load
        page_html = driver.page_source
        
        # Convert HTML to plain text
        soup = BeautifulSoup(page_html, 'html.parser')
        page_text = soup.get_text(separator=' ', strip=True)

        visited_pages[link] = page_text

    driver.quit()

    # Save visited pages as pickle
    with open(TDS_COURSE_PAGE_SCRAPED_FILE, 'wb') as f:
        pickle.dump(visited_pages, f)

    print(f"Scraped and cached {len(visited_pages)} pages to pickle.")
    return visited_pages

def scrape_tds_discource_page():
    """
    Scrape the TDS discourse page to collect all unique links and their rendered HTML content.
    If the content has already been scraped, load it from a pickle file.
    Only collects data from term1, 2025 (1 Jan 2025 - 14 Apr 2025)
    
    Returns:
        Dict[str, str]: A dictionary mapping each visited URL to its rendered HTML content.
    """

    # Logs in, as discourse is locked behind verified student IDs
    def get_logged_in_driver():
        options = Options()

        driver = webdriver.Chrome(options=options)
        
        # Open login page
        driver.get('https://discourse.onlinedegree.iitm.ac.in/login')

        # Fill username and password fields
        wait = WebDriverWait(driver, 10)

        # Wait for <input> with id="login-account-name"
        username_input = wait.until(
            EC.presence_of_element_located((By.XPATH, '//input[@id="login-account-name"]'))
        )
        username_input.send_keys(DISCOURSE_EMAIL)
        
        # Wait for <input> with id="login-account-password"
        password_input = wait.until(
            EC.presence_of_element_located((By.XPATH, '//input[@id="login-account-password"]'))
        )
        password_input.send_keys(DISCOURSE_PASSWORD)
        
        # Submit the form (adjust selector if needed)
        driver.find_element(By.ID, "login-button").click()
        
        # Wait for login to complete — tweak timeout and condition as necessary
        time.sleep(5)  # or use WebDriverWait for better practice
        
        return driver

    # Check if the cache file exists
    if os.path.exists(TDS_DISCOURSE_PAGE_SCRAPED_FILE):
        with open(TDS_DISCOURSE_PAGE_SCRAPED_FILE, 'rb') as f:
            visited_pages = pickle.load(f)
        print(f"Loaded {len(visited_pages)} pages from cached HTML.")
        return visited_pages
    
    # Setup headless browser
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    driver = get_logged_in_driver()

    # Base site
    base_url = "https://discourse.onlinedegree.iitm.ac.in/tags/c/courses/tds-kb/34/term1-2025"

    # Load main page
    driver.get(base_url)
    time.sleep(2)  # wait for JS to load
    print(driver.page_source)

    # Collect links like "#/page1"
    links = set()
    anchors = driver.find_elements(By.TAG_NAME, "a")
    for a in anchors:
        href = a.get_attribute("href")
        if href and "/t/" in href: # /t/ signifies a post on the discourse
            links.add(href)

    # Visit each link
    visited_pages = {}
    for link in links:
        print(f"Visiting: {link}")
        driver.get(link)
        time.sleep(2)  # wait for JS content to load
        page_html = driver.page_source
        
        # Convert HTML to plain text
        soup = BeautifulSoup(page_html, 'html.parser')
        page_text = soup.get_text(separator=' ', strip=True)

        visited_pages[link] = page_text

    driver.quit()

    # Save visited pages as pickle
    with open(TDS_DISCOURSE_PAGE_SCRAPED_FILE, 'wb') as f:
        pickle.dump(visited_pages, f)

    print(f"Scraped and cached {len(visited_pages)} pages to pickle.")
    return visited_pages
