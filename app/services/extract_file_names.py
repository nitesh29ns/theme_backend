from app.db.mongodb import database
from app import config

def extracted_files():
    try:
        collection = database[config.DOCUMENT_COLLECTION]

        file_names = [i['filename'] for i in collection.find()]
     
        return file_names
    
    except Exception as e:
        return "❌ failed: {e}"


