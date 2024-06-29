package extractor.Visitors;

import extractor.Common.Common;
import extractor.FeaturesEntities.Property;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.NullLiteralExpr;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.visitor.TreeVisitor;

import java.util.ArrayList;
import java.util.List;

public class LeavesCollectorVisitor extends TreeVisitor {
    private final ArrayList<Node> m_Leaves = new ArrayList<>();

    @Override
    public void process(Node node) {
        if (node instanceof Comment) {
            return;
        }
        boolean isLeaf = false;
        boolean isGenericParent = isGenericParent(node);
        if (hasNoChildren(node) && isNotComment(node)) {
            if (!node.toString().isEmpty() && (!"null".equals(node.toString()) || (node instanceof NullLiteralExpr))) {
                m_Leaves.add(node);
                isLeaf = true;
            }
        }

        int childId = getChildId(node);
        node.setData(Common.ChildId, childId);
        Property property = new Property(node, isLeaf, isGenericParent);
        node.setData(Common.PropertyKey, property);
    }

    private boolean isGenericParent(Node node) {
        return (node instanceof ClassOrInterfaceType)
                && ((ClassOrInterfaceType) node).getTypeArguments() != null
                && ((ClassOrInterfaceType) node).getTypeArguments().isPresent();
    }

    private boolean hasNoChildren(Node node) {
        return node.getChildNodes().size() == 0;
    }

    private boolean isNotComment(Node node) {
        return !(node instanceof Comment) && !(node instanceof Statement);
    }

    public ArrayList<Node> getLeaves() {
        return m_Leaves;
    }

    private int getChildId(Node node) {
        Node parent = node.getParentNode().get();
        List<Node> parentsChildren = parent.getChildNodes();
        int childId = 0;
        for (Node child : parentsChildren) {
            if (child.getRange().equals(node.getRange())) {
                return childId;
            }
            childId++;
        }
        return childId;
    }
}
