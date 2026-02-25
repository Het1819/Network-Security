
from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://hetpatelias_db_user:<@password>@cluster0.kx7mutd.mongodb.net/?appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)