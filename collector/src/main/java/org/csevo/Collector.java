package org.csevo;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import data.MethodData;
import data.MethodProjectRevision;
import data.ProjectData;
import org.csevo.util.Config;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Paths;

public class Collector {
    
    public static CollectorConfig sConfig;

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.err.println("Exactly one argument, the path to the json config, is required");
            System.exit(-1);
        }

        sConfig = Config.load(Paths.get(args[0]), CollectorConfig.class);
        if (sConfig.collect) {
            MethodDataCollector.collect();
        } else if (sConfig.transform) {
            MethodDataTransformer.transform();
        } else if (sConfig.parse) {
            MethodDataCollector.parseFile(Paths.get(sConfig.javaFile));
        }
    }
    
    public static void log(String msg) {
        if (sConfig.logFile != null) {
            try {
                PrintWriter fos = new PrintWriter(new FileOutputStream(Paths.get(sConfig.logFile).toFile(), true), true);
                fos.println("[" + Thread.currentThread().getId() + "]" + msg);
//                (new Throwable()).printStackTrace(fos);
                fos.close();
            } catch (IOException e) {
                System.err.println("Couldn't log to " + sConfig.logFile);
                System.exit(-1);
            }
        }
    }
    
    public static final Gson GSON;
    public static final Gson GSON_NO_PPRINT;
    static {
        GsonBuilder gsonBuilder = new GsonBuilder()
                .disableHtmlEscaping()
                .serializeNulls()
                .registerTypeAdapter(ProjectData.class, ProjectData.sDeserializer)
                .registerTypeAdapter(MethodData.class, MethodData.sSerDeser)
                .registerTypeAdapter(MethodProjectRevision.class, MethodProjectRevision.sSerializer);
        GSON_NO_PPRINT = gsonBuilder.create();
        gsonBuilder.setPrettyPrinting();
        GSON = gsonBuilder.create();
    }
    
    public static void transform() {
        System.err.println("I'm not implemented yet");
        System.exit(-1);
    }
}
