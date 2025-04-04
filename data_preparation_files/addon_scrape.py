#Imports
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time
import requests
import gdown
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from pptx import Presentation
from docx import Document
from comtypes.client import CreateObject
from win32com.client import Dispatch
load_dotenv("credentials.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")


# Scrape file metadata and links into DF
def get_table(driver, df):
    # Get table values from table_id wb-auto-1
    table = driver.find_element(By.ID, 'wb-auto-1')
    bodys = table.find_element(By.TAG_NAME, 'tbody')
    rows = bodys.find_elements(By.TAG_NAME, 'tr')
    for row in rows:
        year = row.find_element(By.TAG_NAME, 'th').text

        cols = row.find_elements(By.TAG_NAME, 'td')
        for index, col in enumerate(cols):
            # 0 - location
            # 1 - format
            # 2 - title
            # 3 - isic_section
            # 4 - isic_division_author
            # 5 - type
            # 6 - theme
            # 7 - topic
            # 8 - file_link
            match index:
                case 0:
                    location = col.text
                case 1:
                    format = col.text
                case 2:
                    title = col.text
                    #check if contains link
                    try:
                        if col.find_element(By.TAG_NAME, 'a'):
                            file_link = col.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    except:
                        file_link = pd.NA
                case 3:
                    isic_section = col.text
                case 4:
                    isic_division_author = col.text
                case 5:
                    type = col.text
                case 6:
                    theme = col.text
                case 7:
                    topic = col.text
        row_df =  pd.DataFrame([{'year': year, 'location': location, 'format': format, 'title': title, 'isic_section': isic_section, 'isic_division_author': isic_division_author, 'type': type, 'theme': theme, 'topic': topic, 'file_link': file_link}])
        if '.xls' in file_link:
            print('skipping excel file')
        elif '.ppsx' in file_link:
            print('skipping ppsx file')
        else:
            df = pd.concat([df, row_df],ignore_index=True)
    return df

# check if there is next page
def has_next_page(driver):
    next_button = driver.find_element(By.CSS_SELECTOR, "a.paginate_button.next")
    if next_button.is_displayed():
        return True
    else:
        return False

def get_file_name(year,location,index,url):
    file_name = url.split('/')[-1].lower()
    return f"""{year}_{location}_{index}_{file_name}"""

def get_url(file_name):
    splits = file_name.split('_')
    year = splits[0]
    location = splits[1]
    name = splits[2]
    return f'https://www.voorburggroup.org/{year}%20{location}/{name}'

def remove_extension(file_name):
    return file_name.split('.')[0]

# download file
def download_voorburg_file(url, file_path):
    try:
        splits = file_path.split('/')[0:2]
        splits = "/".join(splits)
        os.makedirs(splits)
    except:
        pass
    try:
        r = requests.get(url, allow_redirects=True)
        open(file_path, 'wb').write(r.content)
    except:
        print(f'Error downloading {url}')
        return False
    return True

def download_gdrive(id, file_path):
    try:
        #split file_name by _
        splits = file_path.split('/')[:1]
        for folder_name in splits:
            os.makedirs(folder_name)
    except:
        pass
    try:
        folder_path = "/".join(splits)
        gdown.download(id=id,output=f'{folder_path}/')
    except:
        print(f'Error downloading {id}')
        return False
    return True

def convert_pptx_to_pdf(input_path, output_path):
    powerpoint = CreateObject("PowerPoint.Application")
    powerpoint.Visible = 1
    presentation = powerpoint.Presentations.Open(os.path.abspath(input_path))
    presentation.SaveAs(os.path.abspath(output_path), 32)  # 32 is for PDF format
    presentation.Close()
    powerpoint.Quit()
    print(f"Converted {input_path} to {output_path}")

def convert_docx_to_pdf(input_path, output_path):
    word = Dispatch("Word.Application")
    doc = word.Documents.Open(os.path.abspath(input_path))
    doc.SaveAs(os.path.abspath(output_path), FileFormat=17)  # 17 is for PDF format
    doc.Close()
    word.Quit()
    print(f"Converted {input_path} to {output_path}")

# Code Starts Here
# Set Up Selenium
try:
    driver = webdriver.Chrome()
    driver.get('https://www.voorburggroup.org/papers-eng.htm')
    time.sleep(2)
    print('Selenium setup complete')
except Exception as e:
    print(e)
    print('failed to setup selenium')
    driver.quit()
    quit()

# Scrape links and data from voorburg site
try:
    # Select 100 dropdown
    select = Select(driver.find_element(By.XPATH, '/html/body/main/section[2]/div/div[1]/div[3]/label/select'))
    select.select_by_value('100')
    
    df = pd.DataFrame(columns=['year', 'location', 'format', 'title', 'isic_section', 'isic_division_author', 'type', 'theme', 'topic', 'file_link'])

    while has_next_page(driver):
        df = get_table(driver,df)
        next_button = driver.find_element(By.CSS_SELECTOR, "a.paginate_button.next")
        if next_button.is_displayed():
            next_button.click()
    
    df = df.replace('NA', pd.NA) # replace 'NA' to pd.NA

    df['source_domain'] = df['file_link'].str.extract(r'\.(.*?)\.')
    print('Successfully retrieved data')
except Exception as e:
    print(e)
    print('Failed to scrape data from voorburg site')
    driver.quit()
    quit()

# Add in manuals
try:
    
    row_df =  pd.DataFrame([{'year': '2014', 'location': 'Europe', 'format': '', 'title': 'Eurostat-OECD Methodological Guide for Developing Producer Price Indices for Services', 'isic_section': '', 'isic_division_author': '', 'type': '', 'theme': '', 'topic': 'Manuals', 'file_link': 'https://www.oecd.org/content/dam/oecd/en/publications/reports/2014/12/eurostat-oecd-methodological-guide-for-developing-producer-price-indices-for-services_g1g46f4b/9789264220676-en.pdf', 'source_domain': 'manual'}])
    row_df2 =  pd.DataFrame([{'year': '2020', 'location': 'Singapore', 'format': '', 'title': 'Statistical Best Practices', 'isic_section': '', 'isic_division_author': '', 'type': '', 'theme': '', 'topic': 'Manuals', 'file_link': 'https://www.singstat.gov.sg/-/media/files/standards_and_classifications/sbp2020.ashx', 'source_domain': 'manual'}])
    df = pd.concat([df, row_df, row_df2],ignore_index=True)
    
    print('Successfully added manuals')
except Exception as e:
    print(e)
    
# Compare DF and ChromaDB to find missing files and create new DF

## Load existing Chroma DB
try:
    # embed the paragraph_list into Chroma db
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    chunk_vector_store = Chroma(
        collection_name="chunk_vector_store",
        embedding_function=embeddings,
        persist_directory="./chunk_vector_store",  # Path for Chromadb
    )

    # get list of metadata
    chunk_vs_file_links = [d['file_link'] for d in (chunk_vector_store.get()['metadatas'])]
    filtered_df = df[~df['file_link'].isin(chunk_vs_file_links)]
except Exception as e:
    print(e)
    print('Error in initialising Chroma db')

# Download missing files from new DF
if len(filtered_df) > 0:
    for index, row in filtered_df.iterrows():    
        try:
            if row['file_link'].endswith('.xlsx') or row['file_link'].endswith('.xls'):
                print("skipping excel file")
            else:
                if row['source_domain'] == 'voorburggroup':
                    url = row['file_link']
                    url_file_name = url.split('/')[-1].lower()
                    file_name = f'{row["year"]}_{row["location"]}_{index}_{url_file_name}'
                    folder_name = remove_extension(file_name)
                    path = f'data/{folder_name}/{file_name}'
                    download_voorburg_file(row['file_link'],path)
                    filtered_df.at[index,'pdf_path'] = path
                elif row['source_domain'] == 'google':
                    print(row['file_link'])
                    file_id = row['file_link'].split('/')[-2]
                    folder_name = f'data/{row["year"]}_{row["location"]}_{index}'
                    download_gdrive(file_id,folder_name)
                    filtered_df.at[index,'pdf_path'] = path
                else:
                    print(f'Unknown source domain {row["source_domain"]}')
                    #download file_link
                    try:
                        url = row['file_link']
                        url_file_name = url.split('/')[-1].lower()
                        folder_name = remove_extension(url_file_name)
                        os.makedirs(f'data/{folder_name}', exist_ok=True)
                        
                        if url_file_name.endswith(".ashx"):
                            path = f'data/{folder_name}/{folder_name}.pdf'
                        else:
                            path = f'data/{folder_name}/{url_file_name}'
                        r = requests.get(url, allow_redirects=True)
                        open(path, 'wb').write(r.content) 
                        filtered_df.at[index,'pdf_path'] = path
                    except Exception as e:
                        print(f'Error downloading {url}')
                
                #If file is ppt, convert to pdf
                if row['file_link'].endswith('.pptx') or row['file_link'].endswith('.ppt'):
                    # Create an object of Presentation class
                    convert_pptx_to_pdf(path, f'data/{folder_name}/{folder_name}.pdf')
                    filtered_df.at[index,'pdf_path'] = f'data/{folder_name}/{folder_name}.pdf'
                if row['file_link'].endswith('.docx') or row['file_link'].endswith('.doc'):
                    # Create an object of Docx class
                    convert_docx_to_pdf(path, f'data/{folder_name}/{folder_name}.pdf')
                    filtered_df.at[index,'pdf_path'] = f'data/{folder_name}/{folder_name}.pdf'
        except Exception as e:
            print(e)
            print(f'Failed to download file {row["file_link"]}')

filtered_df.to_csv('addon_scrape.csv',index=False)