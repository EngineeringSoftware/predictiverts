package data;

import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import org.apache.commons.lang3.tuple.Pair;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

public class MethodData {

    public String prjName;
    public int id;
    
    public String name;
    public String returnType;
    public List<Pair<String, String>> params = new ArrayList<>();
    
    public String code;
    
    public String comment;
    public String commentSummary;
    
    public String className;
    public String path;
    
    // Ser+Deser
    public static SerDeser sSerDeser = new SerDeser();
    private static class SerDeser implements JsonSerializer<MethodData>, JsonDeserializer<MethodData> {
        @Override
        public MethodData deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context) throws JsonParseException {
            return sDeserializer.deserialize(json, typeOfT, context);
        }
    
        @Override
        public JsonElement serialize(MethodData src, Type typeOfSrc, JsonSerializationContext context) {
            return sSerializer.serialize(src, typeOfSrc, context);
        }
    }
    
    // Serialization
    public static JsonSerializer<MethodData> sSerializer = getSerializer();
    
    public static JsonSerializer<MethodData> getSerializer() {
        return (obj, type, jsonSerializationContext) -> {
            JsonObject jObj = new JsonObject();
            
            jObj.addProperty("prj_name", obj.prjName);
            jObj.addProperty("id", obj.id);

            jObj.addProperty("name", obj.name);
            jObj.addProperty("return_type", obj.returnType);
            JsonArray aParams = new JsonArray();
            for (Pair<String, String> param : obj.params) {
                JsonArray aParam = new JsonArray();
                aParam.add(param.getLeft());
                aParam.add(param.getRight());
                aParams.add(aParam);
            }
            jObj.add("params", aParams);
            
            jObj.addProperty("code", obj.code);
            
            jObj.addProperty("comment", obj.comment);
            jObj.addProperty("comment_summary", obj.commentSummary);

            jObj.addProperty("class_name", obj.className);
            jObj.addProperty("path", obj.path);
            
            return jObj;
        };
    }
    
    // Deserialization
    public static final JsonDeserializer<MethodData> sDeserializer = getDeserializer();
    
    public static JsonDeserializer<MethodData> getDeserializer() {
        return (json, type, context) -> {
            try {
                MethodData obj = new MethodData();
                
                JsonObject jObj = json.getAsJsonObject();
                obj.prjName = jObj.get("prj_name").getAsString();
                obj.id = jObj.get("id").getAsInt();
                
                obj.name = jObj.get("name").getAsString();
                obj.returnType = jObj.get("return_type").getAsString();
                JsonArray aParams = jObj.getAsJsonArray("params");
                for (JsonElement aParamElem : aParams) {
                    JsonArray aParam = aParamElem.getAsJsonArray();
                    obj.params.add(Pair.of(aParam.get(0).getAsString(), aParam.get(1).getAsString()));
                }

                obj.code = jObj.get("code").getAsString();
                
                obj.comment = jObj.get("comment").getAsString();
                obj.commentSummary = jObj.get("comment_summary").getAsString();
                
                obj.className = jObj.get("class_name").getAsString();
                obj.path = jObj.get("path").getAsString();
                
                return obj;
            } catch (IllegalStateException e) {
                throw new JsonParseException(e);
            }
        };
    }
}
