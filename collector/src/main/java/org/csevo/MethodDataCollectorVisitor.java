package org.csevo;

import com.github.javaparser.ast.body.AnnotationDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.EnumDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.body.TypeDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.javadoc.Javadoc;
import com.github.javaparser.printer.PrettyPrinterConfiguration;
import data.MethodData;
import org.apache.commons.lang3.tuple.Pair;
import org.csevo.util.NLPUtils;

import java.util.LinkedList;
import java.util.List;

public class MethodDataCollectorVisitor extends VoidVisitorAdapter<MethodDataCollectorVisitor.Context> {
    
    private static final int METHOD_LENGTH_MAX = 10_000;
    
    public static class Context {
        String className;
        List<MethodData> methodDataList = new LinkedList<>();
        int ignoredCount = 0;
    }
    
    private static PrettyPrinterConfiguration METHOD_PPRINT_CONFIG = new PrettyPrinterConfiguration();
    static {
        METHOD_PPRINT_CONFIG
                .setPrintJavadoc(false)
                .setPrintComments(false);
    }
    
    @Override
    public void visit(ClassOrInterfaceDeclaration n, Context context) {
        commonVisitTypeDeclaration(n, context);
        super.visit(n, context);
    }
    
    @Override
    public void visit(AnnotationDeclaration n, Context context) {
        commonVisitTypeDeclaration(n, context);
        super.visit(n, context);
    }
    
    @Override
    public void visit(EnumDeclaration n, Context context) {
        commonVisitTypeDeclaration(n, context);
        super.visit(n, context);
    }
    
    public void commonVisitTypeDeclaration(TypeDeclaration<?> n, Context context) {
        // Update context class name
        context.className = n.getNameAsString();
    }

    private String dollaryClassName(TypeDeclaration<?> n) {
        if (n.isNestedType()) {
            return dollaryClassName((TypeDeclaration<?>) n.getParentNode().get());
        }
        return n.getNameAsString();
    }
    
    @Override
    public void visit(MethodDeclaration n, Context context) {
        MethodData methodData = new MethodData();

        if (n.getJavadoc().isPresent()) {
            Javadoc javadoc = n.getJavadoc().get();
            methodData.comment = javadoc.toText();

            // If javadoc is not English or no description part, the summary is empty string
            if (!NLPUtils.isValidISOLatin(methodData.comment) || javadoc.getDescription().toText().trim().length() == 0) {
                methodData.commentSummary = "";
            } else {
                methodData.commentSummary = NLPUtils.getFirstSentence(javadoc.getDescription().toText()).orElse(null);
            }
        }
        else {
            methodData.comment = "";
            methodData.commentSummary = "";
        }

        methodData.code = n.toString(METHOD_PPRINT_CONFIG);
        
        // Ignore if method is too long
//        if (methodData.code.length() > METHOD_LENGTH_MAX) {
//            ++context.ignoredCount;
//            return;
//        }
        
        // Ignore if method is not English
        if (!NLPUtils.isValidISOLatin(methodData.code)) {
            ++context.ignoredCount;
            return;
        }

        try {
            methodData.className = dollaryClassName((TypeDeclaration<?>) n.getParentNode().get());
        }
        catch (Exception e) {
            methodData.className = context.className;
        }
        methodData.className = context.className;

        methodData.name = n.getNameAsString();
        methodData.returnType = n.getType().asString();
        for (Parameter param : n.getParameters()) {
            methodData.params.add(Pair.of(param.getType().asString(), param.getNameAsString()));
        }
        
        context.methodDataList.add(methodData);
    }
}
