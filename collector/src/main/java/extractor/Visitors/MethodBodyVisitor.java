package extractor.Visitors;

import extractor.Common.Common;
import extractor.Common.MethodContent;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import extractor.Common.ExtractArgs;
import static org.csevo.Collector.log;
import java.util.ArrayList;

public class MethodBodyVisitor extends VoidVisitorAdapter<Void> {
    private ArrayList<String> methodBody = new ArrayList<>();

    @Override
    public void visit(MethodDeclaration n, Void arg) {
        /*
         * here you can access the attributes of the method. this method will be called for all methods in this CompilationUnit,
         * including inner class methods
         */
        if (n.getBody().isPresent()) {
            String methodName = n.getName().toString();
            methodBody.add(n.toString().replaceFirst(methodName, "").replace("\n", " "));
        }
        super.visit(n, arg);
    }

    public String getMethodBody() {
        return methodBody.get(0);
    }
}