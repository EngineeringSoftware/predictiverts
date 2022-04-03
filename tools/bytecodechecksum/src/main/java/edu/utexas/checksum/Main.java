package edu.utexas.checksum;

import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.MalformedURLException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.ekstazi.data.RegData;
import org.ekstazi.data.TxtStorer;
import org.ekstazi.hash.BytecodeCleaner;
import org.ekstazi.util.FileUtil;
import org.ekstazi.hash.Hasher;

public class Main {

    public static String currentFolder = System.getProperty("user.dir");
    public static String regDataPath =  System.getProperty("user.dir") + "/" + ".regData";

    public static boolean bytecodeChanged() {
        boolean changed = false;

        FileInputStream inputStream = null;
        try {
            inputStream = new FileInputStream(regDataPath);
        } catch (FileNotFoundException e) {
            // run for the first time, .regData does not exist
            saveRegDataSet(regDataPath, firstRun());
            return true;
        }

        // compare existed regData with original regData
        try {
            // load previous regData from .regData
            Method mExtendedLoad = TxtStorer.class.getDeclaredMethod("extendedLoad", FileInputStream.class);
            mExtendedLoad.setAccessible(true);
            Set<RegData> regDataSet= (Set<RegData>) mExtendedLoad.invoke(new TxtStorer(), inputStream);

            // compare with current regDataSet
            for (RegData regData: new HashSet<>(regDataSet)){
                String newHash = new Hasher(Hasher.Algorithm.CRC32, 1000, true).hashURL(regData.getURLExternalForm());
                if (!newHash.equals(regData.getHash())){
                    changed = true;
                }
                regDataSet.add(new RegData(regData.getURLExternalForm(), newHash));
                regDataSet.remove(regData);
            }

            if(changed){
                saveRegDataSet(regDataPath, regDataSet);
            }

        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            e.printStackTrace();
        }

        return changed;
    }

    private static void saveRegDataSet(String regDatapath, Set<RegData> hashes){
        try {
            Method mExtendedSave = TxtStorer.class.getDeclaredMethod("extendedSave", FileOutputStream.class, Set.class);
            mExtendedSave.setAccessible(true);
            FileOutputStream fos = new FileOutputStream(regDatapath);
            mExtendedSave.invoke(new TxtStorer(), fos, hashes);
        } catch (NoSuchMethodException | FileNotFoundException | InvocationTargetException | IllegalAccessException e1) {
            e1.printStackTrace();
        }
    }

    private static Set<RegData> firstRun(){
        Set<RegData> result = new HashSet<>();
        try{
            final Stream<Path> allPaths = Files.walk(Paths.get(currentFolder));
            List<Path> pathList = allPaths.collect(Collectors.toList());
            for(Path p : pathList){
                // filter the .class file
                if (p.toString().endsWith(".class")){
                    byte[] bytes = new byte[0];
                    try {
                        bytes = FileUtil.loadBytes(p.toFile().toURI().toURL());
                    } catch (MalformedURLException e) {
                        e.printStackTrace();
                    }
                    bytes = BytecodeCleaner.removeDebugInfo(bytes);
                    Method method = null;
                    try {
                        // get the checksum of current file
                        method = Hasher.class.getDeclaredMethod("hashByteArray", byte[].class);
                        method.setAccessible(true);
                        String hash = (String)method.invoke(new Hasher(Hasher.Algorithm.CRC32, 1000, true), bytes);
                        result.add(new RegData(p.toFile().toURI().toURL().toExternalForm(), hash));
                    } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
                        e.printStackTrace();
                    }
                }
            }
        }catch(Exception e){
            e.printStackTrace();
        }
        return result;
    }


    public static void main(String[] args){
        System.out.println(bytecodeChanged());
    }
}
