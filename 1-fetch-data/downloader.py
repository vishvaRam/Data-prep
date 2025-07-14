import os
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from rich import print
import requests
from urllib.parse import urljoin
from selenium.common.exceptions import WebDriverException


class RbiCircularScraper:
    def __init__(self, year,base_url="https://www.rbi.org.in/scripts/bs_circularindexdisplay.aspx  ",
                 download_dir="Downloads"):
        self.year = year
        self.base_url = base_url
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        self.driver = None  # Initialize driver to None
        self.setup_selenium()
        # Setup headers for PDF download
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://www.rbi.org.in/  '
        }

    def setup_selenium(self):
        """Setup Selenium WebDriver."""
        if self.driver is None:  # Only initialize if it's None
            try:
                # Use ChromeDriverManager to handle driver installation and path
                service = Service(ChromeDriverManager().install())
                options = webdriver.ChromeOptions()
                options.add_argument("--headless")  # Remove this line if you want to see the browser
                options.add_argument("--no-sandbox")  # Add this line
                options.add_argument("--disable-dev-shm-usage")  # Add this line
                self.driver = webdriver.Chrome(service=service, options=options)
            except WebDriverException as e:
                print(f"❌ Error setting up Selenium: {e}")
                # Handle or log the error as appropriate for your application
                raise  # Re-raise the exception to stop execution

    def fetch_rbi_page(self):
        """Fetch the RBI circular index page using Selenium."""
        print("🔄 Fetching RBI Circular Index...")
        try:
            self.driver.get(self.base_url)
            print("⏳ Waiting for JavaScript to load...")
            time.sleep(1.5)
            print(f"🖱️ Simulating click for year {self.year}")
            self.driver.execute_script(f"GetYearMonth('{self.year}','0');")
            time.sleep(1.5)  # Wait for data to load
            return self.driver.page_source
        except WebDriverException as e:
            print(f"⚠️ WebDriverException in fetch_rbi_page: {e}")
            # Consider retrying, re-initializing the driver, or exiting
            self.driver.quit()
            self.driver = None
            self.setup_selenium()  # Re-initialize the driver
            return "" # Return empty string to indicate failure.

        except Exception as e:
            print(f"⚠️ Error fetching RBI page: {e}")
            return ""

    def parse_table(self, html):
        """Parse HTML table from RBI circular index page."""
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table', {'class': 'tablebg'})
        rows = []
        headers = ['Circular Number', 'Date Of Issue', 'Department', 'Subject', 'Meant For']
        if not table:
            raise Exception("❌ Failed to find the circular table in HTML")
        for tr in table.find_all('tr')[2:]:  # Skip header rows
            tds = tr.find_all('td')
            if len(tds) == 5:
                circ_num = tds[0].get_text(strip=True)
                date = tds[1].get_text(strip=True)
                dept = tds[2].get_text(strip=True)
                subj = tds[3].get_text(strip=True)
                meant_for = tds[4].get_text(strip=True)
                link_tag = tds[0].find('a')
                link = link_tag['href'] if link_tag else None
                rows.append([circ_num, date, dept, subj, meant_for, link])
        return rows, headers + ['Link']

    def get_pdf_link(self, inner_url):
        """Extract PDF link from inner circular page."""
        full_url = urljoin("https://www.rbi.org.in/scripts/  ", inner_url.lstrip('/'))
        print(full_url)
        try:
            self.driver.get(full_url)
            time.sleep(1.5)
            inner_html = self.driver.page_source
            soup = BeautifulSoup(inner_html, 'html.parser')
            pdf_link_tag = soup.find('a', href=lambda href: href and '.pdf' in href.lower())
            if pdf_link_tag:
                pdf_url = pdf_link_tag['href']
                print(pdf_url)
                return pdf_url
            object_tag = soup.find('object', data=True)
            if object_tag:
                return object_tag['data']
            print("🚫 No PDF found on inner page.")
            return None
        except WebDriverException as e:
            print(f"⚠️ WebDriverException in get_pdf_link: {e}")
            self.driver.quit()
            self.driver = None
            self.setup_selenium()
            return None
        except Exception as e:
            print(f"❌ Error getting PDF link: {e}")
            return None

    def download_pdf(self, pdf_url, circular_number, date_of_issue):
        """Download PDF file and save directly to root download dir with date in filename."""
        safe_circ_number = circular_number.replace('/', '_')
        date_str = date_of_issue.strftime('%Y-%m-%d')
        filename = f"{safe_circ_number}_{date_str}.pdf"
        filepath = os.path.join(self.download_dir, filename)

        if os.path.exists(filepath):
            print(f"📄 PDF already exists: {filename}")
            return

        try:
            full_url = pdf_url if pdf_url.startswith("http") else urljoin("https://www.rbi.org.in/scripts/  ",
                                                                          pdf_url.lstrip('/'))
            print(f"\n📥 Downloading PDF from: {full_url}")

            response = requests.get(full_url, headers=self.headers, stream=True, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')
            if 'application/pdf' not in content_type:
                print(f"⚠️ Warning: Content type is not PDF: {content_type}")

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size = os.path.getsize(filepath)
            if file_size < 1000:
                print(f"⚠️ Warning: Downloaded file is very small ({file_size} bytes)")

            print(f"✅ PDF downloaded: {filepath}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading PDF: {e}")

    def run(self):
        """Main execution method to download all PDFs for the selected year."""
        try:
            html = self.fetch_rbi_page()
            if not html:
                print("❌ Failed to fetch RBI page. Exiting.")
                return []
            rows, headers = self.parse_table(html)
            df = pd.DataFrame(rows, columns=headers)
            df['Date Of Issue'] = pd.to_datetime(df['Date Of Issue'], format='%d.%m.%Y', errors='coerce')
            df = df.dropna(subset=['Date Of Issue']).copy()
            df['Date Of Issue'] = df['Date Of Issue'].dt.strftime('%Y-%m-%d')  # Normalize
            df['Date Of Issue'] = pd.to_datetime(df['Date Of Issue'])
            df_sorted = df.sort_values(by='Date Of Issue', ascending=False)
            df_sorted.to_csv(os.path.join(self.download_dir, "All_Circulars_Dataframe.csv"), index=False)

            downloaded_pdf_paths = set()
            for _, row in df_sorted.iterrows():
                print(f"\n📘 Processing Circular: {row['Circular Number']} - {row['Subject']}")
                inner_page_link = row['Link']
                print(f"\n🔗 Visiting inner page: {inner_page_link}")
                pdf_url = self.get_pdf_link(inner_page_link)
                if pdf_url:
                    date_of_issue = row['Date Of Issue']
                    self.download_pdf(pdf_url, row['Circular Number'], date_of_issue)
                    downloaded_pdf_paths.add(os.path.abspath(self.download_dir))
                else:
                    print(f"🚫 Could not find PDF link for circular: {row['Circular Number']}")
            return list(downloaded_pdf_paths)
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            return []
        finally:
            if self.driver is not None:
                self.driver.quit()  # Ensure driver is quit in finally block

    def test_pdf_download(self, pdf_url):
        """Test function to download a single PDF."""
        print(f"🧪 Testing PDF download from: {pdf_url}")
        test_dir = os.path.join(self.download_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        test_filepath = os.path.join(test_dir, "test.pdf")
        try:
            response = requests.get(pdf_url, headers=self.headers, stream=True, timeout=30)
            response.raise_for_status()
            with open(test_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"✅ Test PDF downloaded: {test_filepath}")
            print(f"📊 File size: {os.path.getsize(test_filepath)} bytes")
            return test_filepath
        except requests.exceptions.RequestException as e:
            print(f"❌ Error in test download: {e}")
            return None
        except Exception as e:
            print(f"❌ Error during test download: {e}")
            return None



if __name__ == "__main__":
    scraper = RbiCircularScraper(2023,download_dir="/workspaces/Data_prep/Code/Data/Raw-pdfs/2023")
    # # Uncomment this to test a specific PDF download before running the full scraper
    # test_pdf = "https://rbidocs.rbi.org.in/rdocs/Notification/PDFs/36NT8C402BE7C2A349E0BFFF3C526668CD7A.PDF  "
    # scraper.test_pdf_download(test_pdf)
    res = scraper.run()
    print(f"\n🏁 Download process completed. PDFs saved in: {res}")