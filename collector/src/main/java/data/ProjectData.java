package data;

import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class ProjectData {

    public String name;
    public String url;
    public List<String> revisions = new LinkedList<>();
    public Map<String, List<String>> parentRevisions = new HashMap<>();
    public Map<String, String> yearRevisions = new HashMap<>();
    
    // Deserialization
    public static final JsonDeserializer<ProjectData> sDeserializer = getDeserializer();
    
    public static JsonDeserializer<ProjectData> getDeserializer() {
        return (json, type, context) -> {
            try {
                ProjectData obj = new ProjectData();
    
                JsonObject jObj = json.getAsJsonObject();
                obj.name = jObj.get("name").getAsString();
                // obj.url = jObj.get("url").getAsString();

                return obj;
            } catch (IllegalStateException e) {
                throw new JsonParseException(e);
            }
        };
    }
}
