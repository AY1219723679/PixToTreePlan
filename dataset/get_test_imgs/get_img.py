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
query = "park"  # Changed from "urban trees" to "urban canopy trees"
save_dir = "unsplash_trees"  # Changed from "dataset" to "unsplash_trees"
num_images = 1000  # Changed from 50 to 100
SCROLL_PAUSE_TIME = 2
use_low_res = False  # Set to False to get high-resolution images

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
    """Extract the highest resolution image URL from an image element"""
    # Check for srcset attribute which contains multiple resolutions
    srcset = img_element.get('srcset')
    if srcset:
        # Parse the srcset attribute properly
        # Format is typically: "url1 size1, url2 size2, ..."
        candidates = []
        parts = [p.strip() for p in srcset.split(',')]
        
        for part in parts:
            if not part:
                continue
                
            # Split by space to separate URL from size descriptor
            subparts = part.strip().split()
            if len(subparts) >= 1:
                url = subparts[0]
                # Extract the width if available (e.g., "1080w")
                width = 0
                if len(subparts) > 1 and subparts[-1].endswith('w'):
                    try:
                        width = int(subparts[-1][:-1])  # Remove 'w' and convert to int
                    except ValueError:
                        pass
                candidates.append((url, width))
        
        if candidates:
            # Sort by width (descending) to get highest resolution first
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]  # Return the URL with the highest width
    
    # Try data-src which often contains high-resolution version
    data_src = img_element.get('data-src')
    if data_src and data_src.startswith('http'):
        return data_src
    
    # Fall back to standard src attribute
    src = img_element.get('src')
    if src and src.startswith('http'):
        return src
            
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
                # Skip small images, thumbnails, logos and profile pictures
                # Check image size attributes if available
                width = img.get('width')
                height = img.get('height')
                
                # Skip images with small dimensions (likely thumbnails, icons, logos)
                if width and height and (int(width) < 300 or int(height) < 300):
                    continue
                
                # Skip profile images by using common profile image URL patterns
                img_class = img.get('class', [])
                alt_text = img.get('alt', '').lower()
                img_id = img.get('id', '').lower()
                
                # Skip images that are likely profile pictures, icons or logos
                if (any(c in ' '.join(img_class) for c in ['avatar', 'profile', 'logo', 'icon', 'thumbnail']) or
                    any(term in alt_text for term in ['profile', 'avatar', 'logo', 'icon', 'user']) or
                    any(term in img_id for term in ['logo', 'icon', 'avatar'])):
                    continue
                
                # Get a suitable image URL (not necessarily high-resolution)
                img_url = extract_image_url(img)
                
                if not img_url or not img_url.startswith('http'):
                    continue
                    
                if "unsplash.com" not in img_url:
                    continue
                
                # Skip images from profile related URLs
                if any(pattern in img_url for pattern in ['/profile/', 'avatar', 'icon', 'logo', 'thumbnail']):
                    continue
                
                # Download image
                try:
                    response = requests.get(img_url, stream=True, timeout=10)
                    if response.status_code != 200:
                        continue
                    
                    # Check image size - skip if too small (thumbnails/icons are usually small)
                    if len(response.content) < 20000:  # Skip images smaller than ~20KB
                        continue
                    
                    # Check for duplicate using content hash
                    content = response.content
                    file_hash = get_file_hash(content)
                    
                    if file_hash in downloaded_files:
                        print("Skipping duplicate image")
                        continue
                    
                    # Save the image
                    downloaded_files.add(file_hash)
                    filename = os.path.join(save_dir, f"park{downloaded+1}.jpg")
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
