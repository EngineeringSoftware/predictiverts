package org.csevo;

import org.csevo.util.Config;
import org.csevo.util.Option;

import java.nio.file.Paths;
import java.nio.file.Path;

public class CollectorConfig extends Config {
    
    // Config
    @Option public boolean collect = false;
    @Option public boolean transform = false;
    @Option public boolean year = false;
    @Option public boolean parse = false;
    
    // Input for collect
    @Option public String projectDir;
    @Option public String projectDataFile;
    @Option public String revision;
    @Option public String javaFile;
    
    // Input for transform
    @Option public String model;
    @Option public String dataType;
    @Option public String dataFile;
    
    // Output
    @Option public String outputDir;
    
    // Log file
    @Option public String logFile;
    
    /**
     * Automatically infers and completes some config values, after loading from file.
     */
    public void autoInfer() {
        if (outputDir != null) {
            Paths.get(outputDir).toFile().mkdirs();
        }
    }
    
    public boolean repOk() {
        if ((collect?1:0) + (transform?1:0) + (parse?1:0)!= 1) {
            return false;
        }
        
        if (collect) {
            if (projectDir == null || !Paths.get(projectDir).toFile().isDirectory()) {
                return false;
            }
            
            if (projectDataFile == null || !Paths.get(projectDataFile).toFile().isFile()) {
                return false;
            }
        }
        
        if (transform) {
            if (model == null || dataType == null || dataFile == null) {
                return false;
            }
        }
        
        if (outputDir == null) {
            return false;
        }
        
        return true;
    }
}
