import os
import time
import requests
import hashlib
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import platform

# Configuration
query = "urban tree"
save_dir = "unsplash_trees"  # Changed from "dataset" to "unsplash_trees"
num_images = 100  # Changed from 50 to 100
SCROLL_PAUSE_TIME = 2
use_low_res = True  # Set to True for standard/lower resolution images that load faster

# Create directory
os.makedirs(save_dir, exist_ok=True)
downloaded_files = set()  # For tracking duplicates

def get_file_hash(file_content):
    """Generate a hash for a file to check for duplicates"""
    return hashlib.md5(file_content).hexdigest()

def setup_driver():
    """Setup and return a Chrome WebDriver"""
    chrome_options = Options()
    # Chrome in headless mode runs faster and doesn't open a visible browser
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        return driver
    except Exception as e:
        print(f"Error setting up Chrome driver: {e}")
        print("If you have Chrome installed in a non-standard location, please provide the path:")
        if platform.system() == "Windows":
            print("Example: C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe")
        else:
            print("Example: /Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        custom_path = input("Chrome path (leave empty to try again with default): ").strip()
        
        if custom_path:
            chrome_options.binary_location = custom_path
            try:
                driver = webdriver.Chrome(
                    service=Service(ChromeDriverManager().install()),
                    options=chrome_options
                )
                return driver
            except Exception as e:
                print(f"Still encountering error: {e}")
                exit(1)

def extract_image_url(img_element):
    """Extract a suitable image URL from an image element based on resolution preference"""
    # Check for srcset attribute which contains multiple resolutions
    srcset = img_element.get('srcset')
    if srcset:
        urls = [part.strip().split(' ')[0] for part in srcset.split(',')]
        if urls:
            if use_low_res:
                return urls[0]  # Get first (lowest resolution) URL when low res is preferred
            else:
                return urls[-1]  # Get last (highest resolution) URL
    
    # Try src attribute (standard resolution)
    src = img_element.get('src')
    if src and src.startswith('http'):
        return src
    
    # Try data-src as fallback
    data_src = img_element.get('data-src')
    if data_src and data_src.startswith('http'):
        return data_src
            
    return None

def main():
    print(f"Starting Unsplash image scraper for '{query}'")
    search_url = f"https://unsplash.com/s/photos/{query.replace(' ', '-')}"
    
    # Initialize Chrome driver
    driver = setup_driver()
    print(f"Opening {search_url}...")
    driver.get(search_url)
    
    # Wait for images to load initially
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, "img"))
        )
    except Exception as e:
        print(f"Error waiting for initial images to load: {e}")
    
    # Scroll to load more images
    print("Scrolling to load more images...")
    scroll_count = 0
    downloaded = 0
    
    while downloaded < num_images and scroll_count < 20:  # Limit scrolling to prevent infinite loops
        # Execute scroll
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        scroll_count += 1
        print(f"Scroll {scroll_count}... ", end="", flush=True)
        
        # Wait for images to load
        time.sleep(SCROLL_PAUSE_TIME)
        print("Done")
        
        # Parse the page with BeautifulSoup to extract image links
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        img_tags = soup.find_all('img')
        
        print(f"Found {len(img_tags)} image tags. Processing...")
        
        for img in img_tags:
            if downloaded >= num_images:
                break
                
            try:
                # Get a suitable image URL (not necessarily high-resolution)
                img_url = extract_image_url(img)
                
                if not img_url or not img_url.startswith('http'):
                    continue
                    
                if "unsplash.com" not in img_url:
                    continue
                
                # Download image
                try:
                    response = requests.get(img_url, stream=True, timeout=10)
                    if response.status_code != 200:
                        continue
                    
                    # Check for duplicate using content hash
                    content = response.content
                    file_hash = get_file_hash(content)
                    
                    if file_hash in downloaded_files:
                        print("Skipping duplicate image")
                        continue
                    
                    # Save the image
                    downloaded_files.add(file_hash)
                    filename = os.path.join(save_dir, f"urban_tree_{downloaded+1}.jpg")
                    with open(filename, "wb") as f:
                        f.write(content)
                    
                    downloaded += 1
                    print(f"Downloaded {downloaded}/{num_images}: {filename}")
                    
                except Exception as e:
                    print(f"Error downloading {img_url}: {e}")
                    
            except Exception as e:
                print(f"Error processing image: {e}")
    
    # Close the driver
    driver.quit()
    
    # Final report
    print(f"\n✅ Download complete! Downloaded {downloaded} images to {os.path.abspath(save_dir)}/")
    print(f"Unique images: {len(downloaded_files)}")

if __name__ == "__main__":
    main()
