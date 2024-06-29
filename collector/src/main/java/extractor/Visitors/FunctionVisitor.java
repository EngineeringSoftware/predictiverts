package extractor.Visitors;

import extractor.Common.Common;
import extractor.Common.MethodContent;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import extractor.Common.ExtractArgs;
import static org.csevo.Collector.log;
import java.util.ArrayList;
import java.util.Arrays;

@SuppressWarnings("StringEquality")
public class FunctionVisitor extends VoidVisitorAdapter<Object> {
    private final ArrayList<MethodContent> m_Methods = new ArrayList<>();
    private final ExtractArgs m_ExtractArgs;

    public FunctionVisitor(ExtractArgs m_extractArgs) {
        this.m_ExtractArgs = m_extractArgs;
    }

    @Override
    public void visit(MethodDeclaration node, Object arg) {
        visitMethod(node);
        super.visit(node, arg);
    }


    private void visitMethod(MethodDeclaration node) {
        extractor.Visitors.LeavesCollectorVisitor leavesCollectorVisitor = new LeavesCollectorVisitor();
        leavesCollectorVisitor.visitPreOrder(node);
        ArrayList<Node> leaves = leavesCollectorVisitor.getLeaves();

        String normalizedMethodName = Common.normalizeName(node.getName().toString(), Common.BlankWord);
        ArrayList<String> splitNameParts = Common.splitToSubtokens(node.getName().toString());
        String splitName = normalizedMethodName;
        if (splitNameParts.size() > 0) {
            splitName = String.join(Common.internalSeparator, splitNameParts);
        }
        splitName = "";
        if (node.getBody().isPresent()) {
            long methodLength = getMethodLength(node.getBody().toString());
            if (m_ExtractArgs.MaxCodeLength > 0) {
                if (methodLength >= m_ExtractArgs.MinCodeLength && methodLength <= m_ExtractArgs.MaxCodeLength) {
                    m_Methods.add(new MethodContent(leaves, splitName));
                }
            } else {
                m_Methods.add(new MethodContent(leaves, splitName));
            }
        }
    }

    private long getMethodLength(String code) {
        String cleanCode = code.replaceAll("\r\n", "\n").replaceAll("\t", " ");
        if (cleanCode.startsWith("{\n"))
            cleanCode = cleanCode.substring(3).trim();
        if (cleanCode.endsWith("\n}"))
            cleanCode = cleanCode.substring(0, cleanCode.length() - 2).trim();
        if (cleanCode.length() == 0) {
            return 0;
        }
        return Arrays.stream(cleanCode.split("\n"))
                .filter(line -> (line.trim() != "{" && line.trim() != "}" && line.trim() != ""))
                .filter(line -> !line.trim().startsWith("/") && !line.trim().startsWith("*")).count();
    }

    public ArrayList<MethodContent> getMethodContents() {
        return m_Methods;
    }
}
