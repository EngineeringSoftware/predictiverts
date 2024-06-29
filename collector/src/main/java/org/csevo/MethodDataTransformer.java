package org.csevo;

import com.github.javaparser.ast.CompilationUnit;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import edu.stanford.nlp.coref.statistical.FeatureExtractor;
import extractor.Common.Common;
import extractor.Common.ExtractArgs;
import extractor.FeaturesEntities.ProgramFeatures;
import extractor.PathExtractor;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.body.MethodDeclaration;
import data.MethodData;
import extractor.Visitors.MethodBodyVisitor;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import static org.csevo.Collector.GSON;
import static org.csevo.Collector.log;
import static org.csevo.Collector.sConfig;

public class MethodDataTransformer {

    public static void transform() throws IOException {
        List<MethodData> methodDataList;
        if (sConfig.dataType.equals("debug")) {
            methodDataList = loadDebugMethodDataList();
        } else {
            // Grab data from MongoDb
            methodDataList = loadMethodDataList();
        }
        log("Processing " + methodDataList.size() + " data");
        final String classPrefix = "public class Test {";
        final String classSuffix = "}";
        final String methodPrefix = "SomeUnknownReturnType f() {";
        final String methodSuffix = "return noSuchReturnValue; }";
        ArrayList<Integer> errorIds = new ArrayList<Integer>();

        switch (sConfig.model) {
            case "Code2Seq":
                int methodIndex = -1;
                ArrayList<ArrayList<ProgramFeatures>> programFeatures = new ArrayList<>();
                FileWriter writer = new FileWriter(sConfig.outputDir + "/" + sConfig.dataType + ".raw.txt");
                for (MethodData m : methodDataList) {
                    methodIndex += 1;
                    String content = classPrefix + m.code + classSuffix;
                    CompilationUnit c;
                    try {
                        c = StaticJavaParser.parse(content);
                    } catch (Exception e) {
                        log("Exception: " + e);
                        errorIds.add(methodIndex);
                        continue;
                    }
                    ExtractArgs m_ExtractArgs = new ExtractArgs(8, 2);
                    PathExtractor pathExtractor = new PathExtractor(m_ExtractArgs);
                    String normalizedName = Common.normalizeName(m.name, Common.BlankWord);
                    ArrayList<String> splitNameParts = Common.splitToSubtokens(m.name);
                    String splitName = normalizedName;
                    if (splitNameParts.size() > 0) {
                        splitName = String.join(Common.internalSeparator, splitNameParts);
                    }
                    try {
                        writer.write(splitName + pathExtractor.extractFeatures(c).get(0) + "\n");
                    } catch (Exception e) {
                        log("Exception: " + e);
                        errorIds.add(methodIndex);
                    }

                }
                writer.close();
                try (FileWriter errWriter = new FileWriter(sConfig.outputDir + "-error-ids.json")) {
                    Gson gson = new GsonBuilder().create();
                    gson.toJson(errorIds, errWriter);
                }
                break;
            case "BiLSTM":
                int indexMethod = -1;
                FileWriter outputWriter = new FileWriter(sConfig.outputDir + "/" + sConfig.dataType + ".raw.txt");
                for (MethodData m : methodDataList) {
                    indexMethod += 1;
                    String content = classPrefix + m.code + classSuffix;
                    CompilationUnit c;
                    try {
                        c = StaticJavaParser.parse(content);
                    } catch (Exception e) {
                        log("Exception: " + e);
                        errorIds.add(indexMethod);
                        continue;
                    }
                    String normalizedName = Common.normalizeName(m.name, Common.BlankWord);
                    ArrayList<String> splitNameParts = Common.splitToSubtokens(m.name);
                    String splitName = normalizedName;
                    if (splitNameParts.size() > 0) {
                        splitName = String.join(Common.internalSeparator, splitNameParts);
                    }
                    try {
                        MethodBodyVisitor visitor = new MethodBodyVisitor();
                        visitor.visit(c, null);
                        outputWriter.write(splitName.replace("|", " ") + "|" + visitor.getMethodBody() + "\n");
                    } catch (Exception e) {
                        log("Exception: " + e);
                        errorIds.add(indexMethod);
                    }
                }
                outputWriter.close();
                try (FileWriter errWriter = new FileWriter(sConfig.outputDir + "-error-ids.json")) {
                    Gson gson = new GsonBuilder().create();
                    gson.toJson(errorIds, errWriter);
                }

                break;
            default:
                System.exit(-1);
                break;
        }
    }

    public static List<MethodData> loadDebugMethodDataList() {
        try {
            return GSON.fromJson(new FileReader(sConfig.dataFile), TypeToken.getParameterized(TypeToken.get(List.class).getType(), TypeToken.get(MethodData.class).getType()).getType());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static List<MethodData> loadMethodDataList() {
        try {
            return GSON.fromJson(new FileReader(sConfig.dataFile), TypeToken.getParameterized(TypeToken.get(List.class).getType(), TypeToken.get(MethodData.class).getType()).getType());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}



