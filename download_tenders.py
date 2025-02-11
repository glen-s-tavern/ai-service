import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid
import requests
import zipfile
import io
import xml.etree.ElementTree as ET
from tqdm import tqdm

import torch
from tqdm_logger_handler import TqdmLoggingHandler
from transformers import AutoTokenizer, AutoModel
from annoy import AnnoyIndex
from logger_config import setup_logger

logger = setup_logger(__name__)

URL = "https://int44.zakupki.gov.ru/eis-integration/services/getDocsIP"
TOKEN = "3735f424-f2ac-4e0b-9c26-7f5b69e5c04a"
SUBSYSTEM_TYPE = "PRIZ"
DOCUMENT_TYPE = "epNotificationEF2020"
OUTPUT_DIR = './tenders_data'
EMBEDDING_DIM = 1024
BATCH_SIZE = 1

def get_tenders_and_contents(url, token, region, subsystem_type, document_type, exact_date):
    time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    idx = uuid.uuid4()
    soap_envelope = f"""
    <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
      <soap:Header><individualPerson_token>{token}</individualPerson_token></soap:Header>
      <soap:Body>
        <ns2:getDocsByOrgRegionRequest xmlns:ns2="http://zakupki.gov.ru/fz44/get-docs-ip/ws">
          <index>
            <id>{idx}</id>
            <createDateTime>{time}</createDateTime>
            <mode>PROD</mode>
          </index>
          <selectionParams>
            <orgRegion>{region}</orgRegion>
            <subsystemType>{subsystem_type}</subsystemType>
            <documentType223>{document_type}</documentType223>
            <periodInfo>
              <exactDate>{exact_date}</exactDate>
            </periodInfo>
          </selectionParams>
        </ns2:getDocsByOrgRegionRequest>
      </soap:Body>
    </soap:Envelope>
    """

    try:
        response = requests.post(url, data=soap_envelope, headers={"Content-Type": "text/xml; charset=utf-8"})

        if response.status_code != 200:
            logger.error(f"HTTP Error: {response.status_code}")
            return {}

        try:
            root = ET.fromstring(response.text)
            archive_urls = [el.text for el in root.findall(".//archiveUrl") if el is not None]

            if not archive_urls:
                logger.warning(f"No archive URLs found for region {region} on {exact_date}")
                return {}

        except ET.ParseError as e:
            logger.error(f"XML Parsing Error: {e}")
            return {}

        headers = {
            "individualPerson_token": token,
            "User-Agent": "PostmanRuntime/7.43.0",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        files_content = {}
        for archive_url in tqdm(archive_urls, desc=f"Downloading archives for {region} on {exact_date}"):
            try:
                archive_response = requests.get(archive_url, headers=headers, stream=True)
                if archive_response.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(archive_response.content)) as z:
                        for file_name in z.namelist():
                            with z.open(file_name) as file:
                                files_content[file_name] = file.read().decode('utf-8', errors='ignore')
                else:
                    logger.warning(f"Failed to download archive {archive_url}: {archive_response.status_code}")
            except Exception as e:
                logger.error(f"Error downloading archive {archive_url}: {e}")

        return files_content

    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        return {}

def get_tenders_names(files_content):
    namespaces = {
        'ns3': 'http://zakupki.gov.ru/oos/export/1',
        'ns5': 'http://zakupki.gov.ru/oos/EPtypes/1'
    }
    return {
        root.find('.//ns5:purchaseNumber', namespaces).text: root.find('.//ns5:purchaseObjectInfo', namespaces).text
        for content in files_content.values()
        for root in [ET.fromstring(content)]
    }

def download_tenders(regions, start_date, end_date):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    all_tenders = {}

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        logger.info(f"Processing date: {date_str}")

        for region in regions:
            logger.info(f"Fetching tenders for region: {region}")

            files_content = get_tenders_and_contents(
                URL, TOKEN, region, SUBSYSTEM_TYPE, DOCUMENT_TYPE, date_str
            )

            if files_content:
                tenders = get_tenders_names(files_content)
                all_tenders.update(tenders)

                region_dir = Path(OUTPUT_DIR) / region / date_str
                region_dir.mkdir(parents=True, exist_ok=True)

                for filename, content in files_content.items():
                    with open(region_dir / filename, 'w', encoding='utf-8') as f:
                        f.write(content)

        current_date += timedelta(days=1)

    with open(Path(OUTPUT_DIR) / 'tenders_summary.json', 'w', encoding='utf-8') as f:
        json.dump(all_tenders, f, ensure_ascii=False, indent=2)

    logger.info(f"Total tenders downloaded: {len(all_tenders)}")

def load_ruroberta_model():
    try:
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruRoberta-large")
        model = AutoModel.from_pretrained("ai-forever/ruRoberta-large")

        # Move model to GPU if available
        model = model.to(device)
        model.eval()

        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Error loading RuRoBERTa model: {e}")
        raise

def generate_embeddings_batch(texts, tokenizer, model, device):
    try:
        inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                           max_length=512, padding=True, add_special_tokens=True)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Use CLS token (first token) instead of mean pooling
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        return None

def create_tender_embeddings(tenders_summary_path, model=None, tokenizer=None, device=None):
    with open(tenders_summary_path, 'r', encoding='utf-8') as f:
        tenders = json.load(f)

    if tokenizer is None or model is None or device is None:
        tokenizer, model, device = load_ruroberta_model()

    annoy_index = AnnoyIndex(EMBEDDING_DIM, 'angular')

    tenders_metadata = {}

    tender_texts = list(tenders.values())
    tender_keys = list(tenders.keys())
    
    logger.info("Building embeddings...")
    for i in tqdm(range(0, len(tender_texts), BATCH_SIZE)):
        batch_texts = tender_texts[i:i+BATCH_SIZE]
        batch_keys = tender_keys[i:i+BATCH_SIZE]

        embeddings = generate_embeddings_batch(batch_texts, tokenizer, model, device)

        if embeddings is not None:
            for j, (embedding, key, text) in enumerate(zip(embeddings, batch_keys, batch_texts)):
                index = i + j
                annoy_index.add_item(index, embedding)

                tenders_metadata[index] = {
                    'purchase_number': key,
                    'tender_name': text
                }

    annoy_index.build(10)
    annoy_index.save(f'{OUTPUT_DIR}/tenders_index.ann')

    with open(f'{OUTPUT_DIR}/tenders_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(tenders_metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Created ANNOY index with {len(tenders_metadata)} tenders")
    logger.info(f"ANNOY index and metadata saved in {OUTPUT_DIR}")

def main():
    parser = argparse.ArgumentParser(description='Download and process tenders')
    parser.add_argument('--regions', nargs='+', default=['77'], help='List of region codes')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--vectorize', action='store_true', help='Create vector embeddings')

    args = parser.parse_args()

    download_tenders(args.regions, args.start_date, args.end_date)

    if args.vectorize:
        create_tender_embeddings(f'{OUTPUT_DIR}/tenders_summary.json')

if __name__ == '__main__':
    main() 
