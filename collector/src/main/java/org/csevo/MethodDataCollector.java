package org.csevo;

import com.github.javaparser.ParseProblemException;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.google.gson.stream.JsonWriter;
import data.MethodData;
import data.MethodProjectRevision;
import data.ProjectData;
import org.csevo.util.BashUtils;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import static org.csevo.Collector.GSON;
import static org.csevo.Collector.GSON_NO_PPRINT;
import static org.csevo.Collector.log;
import static org.csevo.Collector.sConfig;

public class MethodDataCollector {
    
    private static ProjectData sProjectData;
    
    private static Map<Integer, Integer> sMethodDataIdHashMap = new HashMap<>();
    private static int sCurrentMethodDataId = 0;
    private static List<MethodProjectRevision> sMethodProjectRevisionList = new LinkedList<>();
    private static Map<Integer, List<Integer>> sFileCache = new HashMap<>();
    
    private static JsonWriter sMethodDataWriter;
    private static JsonWriter sMethodProjectRevisionWriter;
    
    public static void collect() {
        try {
            // 1. Load project data
            sProjectData = GSON.fromJson(new FileReader(Paths.get(sConfig.projectDataFile).toFile()), ProjectData.class);

            // 2. Init the writers for saving
            sMethodDataWriter = GSON.newJsonWriter(new FileWriter(sConfig.outputDir + "/method-data.json"));
            sMethodDataWriter.beginArray();
            // 3. collect method
            collectMethod();
            
            // -1. Close readers
            sMethodDataWriter.endArray();
            sMethodDataWriter.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /** This method parse the given java file and revision, and extract all the methods in that file. */
    public static void parseFile(Path javaFile) {
        try {
            // 1. Load project data
            sProjectData = GSON.fromJson(new FileReader(Paths.get(sConfig.projectDataFile).toFile()), ProjectData.class);

            // 2. Init the writers for saving
            sMethodDataWriter = GSON.newJsonWriter(new FileWriter(sConfig.outputDir + "/" + javaFile.getFileName().toString()
                    + "-method-data-" + sConfig.revision + ".json"));
            sMethodDataWriter.beginArray();
            // 3. collect method
            collectMethod(javaFile);

            // -1. Close readers
            sMethodDataWriter.endArray();
            sMethodDataWriter.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /** This method parse all the methods in the given java class. */
    public static void collectMethod(Path javaFile) throws  IOException{
        Path projectPath = Paths.get(sConfig.projectDir);

        // Parse java file and get methods
        MethodDataCollectorVisitor visitor = new MethodDataCollectorVisitor();
        int parseErrorCount = 0;
        int ignoredCount = 0;
        int reuseFileCount = 0;
        int parseFileCount = 0;

	String path = javaFile.toString();

        MethodDataCollectorVisitor.Context context = new MethodDataCollectorVisitor.Context();
        try {
            CompilationUnit cu = StaticJavaParser.parse(javaFile);
            cu.accept(visitor, context);
        } catch (ParseProblemException e) {
            ++parseErrorCount;
        }

        ignoredCount += context.ignoredCount;

        for (MethodData methodData : context.methodDataList) {
            methodData.path = path;
            int methodId = addMethodData(methodData);
        }


        ++parseFileCount;

        log("Parsed " + parseFileCount + " files. " +
                "Reused " + reuseFileCount + " files. " +
                "Parsing error for " + parseErrorCount + " files. " +
                "Ignored " + ignoredCount + " methods. " +
                "Total collected " + sMethodDataIdHashMap.size() + " methods.");
    }

    private static void collectMethod() throws IOException {
        // Find all java files
        Path projectPath = Paths.get(sConfig.projectDir);
        List<Path> javaFiles = Files.walk(projectPath)
                .filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".java"))
                .sorted(Comparator.comparing(Object::toString))
                .collect(Collectors.toList());
        log("Got " + javaFiles.size() + " files to parse");

        // For each java file, parse and get methods
        MethodDataCollectorVisitor visitor = new MethodDataCollectorVisitor();
        int parseErrorCount = 0;
        int ignoredCount = 0;
        int reuseFileCount = 0;
        int parseFileCount = 0;
        for (Path javaFile : javaFiles) {
            int fileHash = getFileHash(javaFile);
            List<Integer> idsFile = sFileCache.get(fileHash);
	    if (idsFile == null) {
		idsFile = new LinkedList<>();
		String path = projectPath.relativize(javaFile).toString();

		MethodDataCollectorVisitor.Context context = new MethodDataCollectorVisitor.Context();
		try {
		    CompilationUnit cu = StaticJavaParser.parse(javaFile);
		    cu.accept(visitor, context);
		} catch (ParseProblemException e) {
		    ++parseErrorCount;
		}

		ignoredCount += context.ignoredCount;

		for (MethodData methodData : context.methodDataList) {
		    methodData.path = path;
		    int methodId = addMethodData(methodData);
		    idsFile.add(methodId);
		}

		// Update file cache
		sFileCache.put(fileHash, idsFile);
		++parseFileCount;
	    } else {
		++reuseFileCount;
	    }
	}
        log("Parsed " + parseFileCount + " files. " +
                "Reused " + reuseFileCount + " files. " +
                "Parsing error for " + parseErrorCount + " files. " +
                "Ignored " + ignoredCount + " methods. " +
                "Total collected " + sMethodDataIdHashMap.size() + " methods.");
    }

    private static int getFileHash(Path javaFile) throws IOException {
        // Hash both the path and the content
        return Objects.hash(javaFile.toString(), Arrays.hashCode(Files.readAllBytes(javaFile)));
    }
    
    private static int addMethodData(MethodData methodData) {
        // Don't duplicate previous appeared methods (keys: path, code, comment)
        int hash = Objects.hash(methodData.path, methodData.code, methodData.comment);
        Integer prevMethodDataId = sMethodDataIdHashMap.get(hash);
        if (prevMethodDataId != null) {
            // If this method data already existed before, retrieve its id
            return prevMethodDataId;
        } else {
            // Allocate a new id and save this data to the hash map
            methodData.id = sCurrentMethodDataId;
            methodData.prjName = sProjectData.name;
            ++sCurrentMethodDataId;
            sMethodDataIdHashMap.put(hash, methodData.id);
            
            // Save the method data
            GSON.toJson(methodData, MethodData.class, sMethodDataWriter);
            return methodData.id;
        }
    }
}
