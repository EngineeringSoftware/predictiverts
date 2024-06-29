package extractor;

import com.github.javaparser.ast.body.MethodDeclaration;
import extractor.Common.Common;
import extractor.Common.MethodContent;
import extractor.FeaturesEntities.ProgramFeatures;
import extractor.FeaturesEntities.Property;
import extractor.Common.ExtractArgs;
import extractor.Visitors.FunctionVisitor;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import static org.csevo.Collector.log;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@SuppressWarnings("StringEquality")
public class PathExtractor {
    private final static String upSymbol = "|";
    private final static String downSymbol = "|";
    private static final Set<String> s_ParentTypeToAddChildId = Stream
            .of("AssignExpr", "ArrayAccessExpr", "FieldAccessExpr", "MethodCallExpr")
            .collect(Collectors.toCollection(HashSet::new));
    private final ExtractArgs m_ExtractArgs;

    public PathExtractor(ExtractArgs m_ExtractArgs) {
        this.m_ExtractArgs =  m_ExtractArgs;
    }

    private static ArrayList<Node> getTreeStack(Node node) {
        ArrayList<Node> upStack = new ArrayList<>();
        Node current = node;
        while (current != null) {
            upStack.add(current);
            if (current.getParentNode().isPresent()) {
                current = current.getParentNode().get();
            }
            else {
                current = null;
            }
        }
        return upStack;
    }

    public ArrayList<ProgramFeatures> extractFeatures(CompilationUnit m_Declaration) {
        FunctionVisitor functionVisitor = new FunctionVisitor(m_ExtractArgs);
        functionVisitor.visit(m_Declaration, null);
        ArrayList<MethodContent> methods = functionVisitor.getMethodContents();
        return generatePathFeatures(methods);
    }

    /*
    private CompilationUnit parseFileWithRetries(String code) {
        final String classPrefix = "public class Test {";
        final String classSuffix = "}";
        final String methodPrefix = "SomeUnknownReturnType f() {";
        final String methodSuffix = "return noSuchReturnValue; }";

        String content = code;
        CompilationUnit parsed;
        try {
            parsed = JavaParser.parse(content);
        } catch (ParseProblemException e1) {
            // Wrap with a class and method
            try {
                content = classPrefix + methodPrefix + code + methodSuffix + classSuffix;
                parsed = JavaParser.parse(content);
            } catch (ParseProblemException e2) {
                // Wrap with a class only
                content = classPrefix + code + classSuffix;
                parsed = JavaParser.parse(content);
            }
        }

        return parsed;
    }
*/
    private ArrayList<ProgramFeatures> generatePathFeatures(ArrayList<MethodContent> methods) {
        ArrayList<ProgramFeatures> methodsFeatures = new ArrayList<>();
        for (MethodContent content : methods) {
            ProgramFeatures singleMethodFeatures = generatePathFeaturesForFunction(content);
            if (!singleMethodFeatures.isEmpty()) {
                methodsFeatures.add(singleMethodFeatures);
            }
        }
        return methodsFeatures;
    }

    private ProgramFeatures generatePathFeaturesForFunction(MethodContent methodContent) {
        ArrayList<Node> functionLeaves = methodContent.getLeaves();
        ProgramFeatures programFeatures = new ProgramFeatures(methodContent.getName());

        for (int i = 0; i < functionLeaves.size(); i++) {
            for (int j = i + 1; j < functionLeaves.size(); j++) {
                String separator = Common.EmptyString;

                String path = generatePath(functionLeaves.get(i), functionLeaves.get(j), separator);
                if (path != Common.EmptyString) {
                    Property source = functionLeaves.get(i).getData(Common.PropertyKey);
                    Property target = functionLeaves.get(j).getData(Common.PropertyKey);
                    programFeatures.addFeature(source, path, target);
                }
            }
        }
        return programFeatures;
    }

    private String generatePath(Node source, Node target, String separator) {

        StringJoiner stringBuilder = new StringJoiner(separator);
        ArrayList<Node> sourceStack = getTreeStack(source);
        ArrayList<Node> targetStack = getTreeStack(target);

        int commonPrefix = 0;
        int currentSourceAncestorIndex = sourceStack.size() - 1;
        int currentTargetAncestorIndex = targetStack.size() - 1;
        while (currentSourceAncestorIndex >= 0 && currentTargetAncestorIndex >= 0
                && sourceStack.get(currentSourceAncestorIndex) == targetStack.get(currentTargetAncestorIndex)) {
            commonPrefix++;
            currentSourceAncestorIndex--;
            currentTargetAncestorIndex--;
        }

        int pathLength = sourceStack.size() + targetStack.size() - 2 * commonPrefix;
        if (pathLength > m_ExtractArgs.MaxPathLength) {
            return Common.EmptyString;
        }

        if (currentSourceAncestorIndex >= 0 && currentTargetAncestorIndex >= 0) {
            int pathWidth = targetStack.get(currentTargetAncestorIndex).getData(Common.ChildId)
                    - sourceStack.get(currentSourceAncestorIndex).getData(Common.ChildId);
            if (pathWidth > m_ExtractArgs.MaxPathWidth) {
                return Common.EmptyString;
            }
        }

        for (int i = 0; i < sourceStack.size() - commonPrefix; i++) {
            Node currentNode = sourceStack.get(i);
            String childId = Common.EmptyString;
            String parentRawType = currentNode.getParentNode().get().getData(Common.PropertyKey).getRawType();
            if (i == 0 || s_ParentTypeToAddChildId.contains(parentRawType)) {
                childId = saturateChildId(currentNode.getData(Common.ChildId))
                        .toString();
            }
            stringBuilder.add(String.format("%s%s%s",
                    currentNode.getData(Common.PropertyKey).getType(true), childId, upSymbol));
        }

        Node commonNode = sourceStack.get(sourceStack.size() - commonPrefix);
        String commonNodeChildId = Common.EmptyString;
        Property parentNodeProperty;
        if (commonNode.getParentNode().get().containsData(Common.PropertyKey)) {
            parentNodeProperty = commonNode.getParentNode().get().getData(Common.PropertyKey);
        }
        else {
            parentNodeProperty = null;
        }
        String commonNodeParentRawType = Common.EmptyString;
        if (parentNodeProperty != null) {
            commonNodeParentRawType = parentNodeProperty.getRawType();
        }
        if (s_ParentTypeToAddChildId.contains(commonNodeParentRawType)) {
            commonNodeChildId = saturateChildId(commonNode.getData(Common.ChildId))
                    .toString();
        }
        stringBuilder.add(String.format("%s%s",
                commonNode.getData(Common.PropertyKey).getType(true), commonNodeChildId));

        for (int i = targetStack.size() - commonPrefix - 1; i >= 0; i--) {
            Node currentNode = targetStack.get(i);
            String childId = Common.EmptyString;
            if (i == 0 || s_ParentTypeToAddChildId.contains(currentNode.getData(Common.PropertyKey).getRawType())) {
                childId = saturateChildId(currentNode.getData(Common.ChildId))
                        .toString();
            }
            stringBuilder.add(String.format("%s%s%s", downSymbol,
                    currentNode.getData(Common.PropertyKey).getType(true), childId));
        }

        return stringBuilder.toString();
    }

    private Integer saturateChildId(int childId) {
        return Math.min(childId, m_ExtractArgs.MaxChildId);
    }
}
