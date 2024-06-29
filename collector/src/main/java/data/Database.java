package data;

import com.mongodb.MongoClient;
import com.mongodb.MongoClientURI;
import com.mongodb.ServerAddress;

import com.mongodb.client.MongoDatabase;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.model.IndexOptions;
import com.mongodb.client.model.Indexes;

public class Database {
    private String mongodb_url;
    private MongoClient mongoClient;
    private MongoDatabase database;


    public Database(boolean local) {
        if (local) {
            mongodb_url = "mongodb://127.0.0.1:20144";
        }
        else {
            mongodb_url = "mongodb://luzhou.ece.utexas.edu:20144";
        }
        // Connect to database
        MongoClientURI connectionString = new MongoClientURI(mongodb_url);
        MongoClient mongoClient = new MongoClient(connectionString);

        init_db();
    }

    private MongoDatabase getDatabase() {
        return mongoClient.getDatabase("data");
    }

    private MongoCollection<org.bson.Document> clProjectData() {
        return getDatabase().getCollection("ProjectData");
    }

    private MongoCollection<org.bson.Document> clMethodData() {
        return getDatabase().getCollection("MethodData");
    }

    private MongoCollection<org.bson.Document> clMethodProjectRevision() {
        return getDatabase().getCollection("MethodProjectRevision");
    }
    private void init_db() {
        IndexOptions indexOptions = new IndexOptions().unique(true).name("key");
        clProjectData().createIndex(Indexes.ascending("name"), indexOptions);
        clMethodData().createIndex(Indexes.ascending("prj_name", "id"), indexOptions);
        clMethodProjectRevision().createIndex(Indexes.ascending("prj_name", "revision"), indexOptions);
    }
}
