import sys
import pymongo
mongo = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo["financial_rag"]
col = db["raw_chunks"]
print(f"Mongo raw_chunks count: {col.count_documents({})}")
mongo.close()
