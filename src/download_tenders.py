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
from annoy import AnnoyIndex
from src.logger_config import setup_logger
from src.models import ModelFactory, ModelType
from src.database import Database
import os

logger = setup_logger(__name__)

URL = "https://int44.zakupki.gov.ru/eis-integration/services/getDocsIP"
TOKEN = "3735f424-f2ac-4e0b-9c26-7f5b69e5c04a"
SUBSYSTEM_TYPE = "PRIZ"
DOCUMENT_TYPE = "epNotificationEF2020"
OUTPUT_DIR = 'resources/tenders_data'
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
        response.raise_for_status()  # Проверяем статус ответа

        if not response.text.strip():  # Проверяем, что ответ не пустой
            logger.warning(f"Empty response for region {region} on {exact_date}")
            return {}

        try:
            root = ET.fromstring(response.text)
            archive_urls = [el.text for el in root.findall(".//archiveUrl") if el is not None and el.text]

            if not archive_urls:
                logger.warning(f"No archive URLs found for region {region} on {exact_date}")
                return {}

        except ET.ParseError as e:
            logger.error(f"XML Parsing Error for region {region} on {exact_date}: {e}")
            logger.debug(f"Response text: {response.text[:200]}...")  # Логируем начало ответа
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
        for archive_url in archive_urls:
            try:
                archive_response = requests.get(archive_url, headers=headers, stream=True)
                archive_response.raise_for_status()

                if not archive_response.content:  # Проверяем, что архив не пустой
                    logger.warning(f"Empty archive content from {archive_url}")
                    continue

                with zipfile.ZipFile(io.BytesIO(archive_response.content)) as z:
                    for file_name in z.namelist():
                        with z.open(file_name) as file:
                            content = file.read().decode('utf-8', errors='ignore')
                            if content.strip():  # Проверяем, что содержимое файла не пустое
                                files_content[file_name] = content

            except (requests.RequestException, zipfile.BadZipFile, UnicodeDecodeError) as e:
                logger.error(f"Error processing archive {archive_url}: {e}")
                continue

        return files_content

    except requests.RequestException as e:
        logger.error(f"Request failed for region {region} on {exact_date}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error for region {region} on {exact_date}: {e}")
        return {}

import xml.etree.ElementTree as ET
import logging

def get_tenders_info(files_content):
    """Извлекает информацию о тендерах из XML файлов."""
    namespaces = {
        'ns2': 'http://zakupki.gov.ru/oos/base/1',
        'ns3': 'http://zakupki.gov.ru/oos/export/1',
        'ns4': 'http://zakupki.gov.ru/oos/common/1',
        'ns5': 'http://zakupki.gov.ru/oos/EPtypes/1',
        'ns6': 'http://zakupki.gov.ru/oos/common/1',
        'ns8': 'http://zakupki.gov.ru/oos/types/1'
    }
    
    tenders_info = {}
    
    for content in files_content.values():
        try:
            root = ET.fromstring(content)
            
            # Номер тендера
            purchase_number = root.find('.//ns5:purchaseNumber', namespaces)
            if purchase_number is None or not purchase_number.text:
                logger.warning("Skipping tender - no purchase number found")
                continue
            purchase_number = purchase_number.text.strip()
            
            # Название тендера
            purchase_object = root.find('.//ns5:purchaseObjectInfo', namespaces)
            if purchase_object is None:
                purchase_object = root.find('.//ns5:notificationInfo/ns5:purchaseObjectInfo', namespaces)
            if purchase_object is None or not purchase_object.text:
                logger.warning(f"Skipping tender {purchase_number} - no name found")
                continue
            purchase_name = purchase_object.text.strip()
            
            # Цена
            price = None
            price_element = root.find('.//ns5:maxPrice', namespaces)
            if price_element is not None and price_element.text:
                try:
                    price = float(price_element.text.replace(',', '.'))
                except ValueError:
                    logger.warning(f"Could not parse price for tender {purchase_number}")

            # Даты
            publish_date = None
            publish_element = root.find('.//ns5:notificationInfo/ns5:procedureInfo/ns5:collectingInfo/ns5:startDT', namespaces)
            if publish_element is not None:
                publish_date = publish_element.text.strip() if publish_element.text else None
            
            update_date = None
            update_element = root.find('.//ns6:modificationDate', namespaces)
            if update_element is not None:
                update_date = update_element.text.strip() if update_element.text else None
            
            end_date = None
            end_element = root.find('.//ns5:notificationInfo/ns5:procedureInfo/ns5:collectingInfo/ns5:endDT', namespaces)
            if end_element is not None:
                end_date = end_element.text.strip() if end_element.text else None
            
            # Тип закона
            law_type = None
            law_element = root.find('.//ns5:preferensesInfo/ns5:preferenseInfo/ns4:preferenseRequirementInfo/ns2:name', namespaces)
            if law_element is not None:
                law_type = law_element.text.strip() if law_element.text else None

            # ОКПД2 - try multiple possible locations
            okpd2_code = None
            # First try: standard location
            okpd2_element = root.find('.//ns5:notDrugPurchaseObjectsInfo/ns4:purchaseObject/ns4:OKPD2/ns2:OKPDCode', namespaces)
            if okpd2_element is None:
                # Second try: alternative location
                okpd2_element = root.find('.//ns5:purchaseObjectsInfo/ns5:notDrugPurchaseObjectsInfo/ns4:purchaseObject/ns4:OKPD2/ns2:OKPDCode', namespaces)
            if okpd2_element is None:
                # Third try: another possible location
                okpd2_element = root.find('.//ns5:purchaseObjects/ns4:purchaseObject/ns4:OKPD2/ns2:OKPDCode', namespaces)
            if okpd2_element is None:
                # Fourth try: fallback location
                okpd2_element = root.find('.//ns4:OKPD2/ns2:OKPDCode', namespaces)
            
            if okpd2_element is not None:
                okpd2_code = okpd2_element.text.strip() if okpd2_element.text else None

            # ИНН заказчика
            customer_inn = None
            inn_element = root.find('.//ns5:purchaseResponsibleInfo/ns5:responsibleOrgInfo/ns5:INN', namespaces)
            if inn_element is not None:
                customer_inn = inn_element.text.strip() if inn_element.text else None
            
            # Название заказчика
            customer_name = None
            name_element = root.find('.//ns5:purchaseResponsibleInfo/ns5:responsibleOrgInfo/ns5:fullName', namespaces)
            if name_element is not None:
                customer_name = name_element.text.strip() if name_element.text else None

            # Способ закупки
            purchase_method = None
            method_element = root.find('.//ns5:commonInfo/ns5:placingWay/ns2:name', namespaces)
            if method_element is not None:
                purchase_method = method_element.text.strip() if method_element.text else None

            # Отладочный вывод
            #if law_type is None or okpd2_code is None:
                #print(f"Could not find law_type or okpd2_code for tender {purchase_number}")
                #print("XML Structure:")
                #for elem in root.iter():
                    #print(f"{elem.tag}: {elem.text}")

            tenders_info[purchase_number] = {
                'name': purchase_name,
                'price': price,
                'publish_date': publish_date,
                'update_date': update_date,
                'end_date': end_date,
                'law_type': law_type,
                'purchase_method': purchase_method,
                'okpd2_code': okpd2_code,
                'customer_inn': customer_inn,
                'customer_name': customer_name
            }
            
        except ET.ParseError as e:
            logger.error(f"Error parsing XML content: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing tender: {e}")
            continue
    
    return tenders_info


def download_tenders(regions, start_date, end_date, save_xml):
    db = Database()
    
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    total_days = (end_date - current_date).days + 1
    
    for current_date in tqdm(
        [current_date + timedelta(days=x) for x in range(total_days)],
        desc="Processing dates"
    ):
        date_str = current_date.strftime('%Y-%m-%d')

        for region in regions:
            existing_tenders = db.get_tenders_by_region_date(region, date_str)
            if existing_tenders:
                logger.info(f"Region {region} date {date_str} already exists, skipping...")
                continue

            logger.info(f"Downloading tenders for region {region} date {date_str}")
            files_content = get_tenders_and_contents(
                URL, TOKEN, region, SUBSYSTEM_TYPE, DOCUMENT_TYPE, date_str
            )
            

            if files_content:
                # Save raw XML files
                if save_xml:
                    os.makedirs(f'{OUTPUT_DIR}/raw_xml', exist_ok=True)
                    for file_name, content in files_content.items():
                        with open(f'{OUTPUT_DIR}/raw_xml/{region}_{date_str}_{file_name}', 'w', encoding='utf-8') as f:
                            f.write(content)

                tenders_info = get_tenders_info(files_content)

                date_tenders = [
                    {
                        "id": tender_id,
                        "name": tender_info['name'],
                        "price": tender_info['price'],
                        "law_type": tender_info['law_type'],
                        "purchase_method": tender_info['purchase_method'],
                        "okpd2_code": tender_info['okpd2_code'],
                        "publish_date": tender_info['publish_date'],
                        "end_date": tender_info['end_date'],
                        "customer_inn": tender_info['customer_inn'],
                        "customer_name": tender_info['customer_name']
                    }
                    for tender_id, tender_info in tenders_info.items()
                ]

                inserted_count = db.insert_tenders(date_tenders, region, date_str)
                logger.info(f"Inserted {inserted_count} tenders for region {region} date {date_str}")

def create_tender_embeddings(model_type: ModelType = ModelType.roberta):
    try:
        db = Database()
        model = ModelFactory.get_model(model_type)
        
        # Получаем все тендеры из базы
        tenders = db.get_all_tenders()
        
        tender_texts = list(tenders.values())
        tender_keys = list(tenders.keys())

        logger.info(f"Building embeddings using {model.__class__.__name__}...")
        embeddings = model.get_embeddings(tender_texts)

        if embeddings is None:
            raise ValueError("Failed to generate embeddings")

        # Создаем и сохраняем индекс
        annoy_index = AnnoyIndex(model.embedding_dim, 'angular')
        tenders_metadata = {}

        for i, (embedding, key, text) in enumerate(zip(embeddings, tender_keys, tender_texts)):
            annoy_index.add_item(i, embedding)
            tenders_metadata[i] = {
                'purchase_number': key,
                'tender_name': text
            }

        annoy_index.build(10)
        model_name = model.__class__.__name__.lower().replace('model', '')
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        annoy_index.save(f'{OUTPUT_DIR}/tenders_index_{model_name}.ann')

        with open(f'{OUTPUT_DIR}/tenders_metadata_{model_name}.json', 'w', encoding='utf-8') as f:
            json.dump(tenders_metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Created ANNOY index with {len(tenders_metadata)} tenders")
        return True
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download and process tenders')
    parser.add_argument('--regions', nargs='+', default=['77'], help='List of region codes')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--vectorize', action='store_true', help='Create vector embeddings')
    parser.add_argument('--model-type', type=str, choices=['roberta', 'fasttext'],
                       default='roberta', help='Model type for vectorization (roberta or fasttext)')
    parser.add_argument('--save_xml', action='store_true', help='Save source xml to disk')

    args = parser.parse_args()

    all_regions = range(1, 100)
    all_regions = [str(el) for el in all_regions]
    download_tenders(args.regions, args.start_date, args.end_date, args.save_xml)

    if args.vectorize:
        create_tender_embeddings(
            model_type=ModelType(args.model_type)
        )

if __name__ == '__main__':
    main()
