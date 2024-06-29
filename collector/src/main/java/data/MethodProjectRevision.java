package data;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializer;

import java.util.LinkedList;
import java.util.List;

public class MethodProjectRevision {
    
    public String prjName;
    public String revision;
    public List<Integer> methodIds = new LinkedList<>();
    public String year = "null";
    
    // Serialization
    public static JsonSerializer<MethodProjectRevision> sSerializer = getSerializer();
    
    public static JsonSerializer<MethodProjectRevision> getSerializer() {
        return (obj, type, jsonSerializationContext) -> {
            JsonObject jObj = new JsonObject();
            
            jObj.addProperty("prj_name", obj.prjName);
            jObj.addProperty("revision", obj.revision);
            jObj.addProperty("year", obj.year);

            JsonArray aMethodIds = new JsonArray();
            for (int methodId : obj.methodIds) {
                aMethodIds.add(methodId);
            }
            jObj.add("method_ids", aMethodIds);
            
            return jObj;
        };
    }
}
