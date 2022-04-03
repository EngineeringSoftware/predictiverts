import java.util.Collections;

import org.apache.maven.AbstractMavenLifecycleParticipant;
import org.apache.maven.MavenExecutionException;
import org.apache.maven.execution.MavenSession;
import org.apache.maven.model.Dependency;
import org.apache.maven.model.PluginConfiguration;
import org.apache.maven.model.PluginExecution;
import org.codehaus.plexus.component.annotations.Component;
import org.apache.maven.project.MavenProject;
import org.apache.maven.model.Plugin;
import org.apache.maven.plugin.MojoFailureException;
import org.codehaus.plexus.util.xml.Xpp3Dom;

@Component( role = AbstractMavenLifecycleParticipant.class, hint = "pit")
public class PitExtension extends AbstractMavenLifecycleParticipant{

    @Override
    public void afterProjectsRead(MavenSession session) throws MavenExecutionException {
        try {
            configurePIT(session);
        } catch (MojoFailureException mfe) {
            throw new MavenExecutionException("Unable to install PIT", mfe);
        }
    }

    @Override
    public void afterSessionStart(MavenSession session) throws MavenExecutionException {
        super.afterSessionStart(session);
    }

    @Override
    public void afterSessionEnd(MavenSession session) throws MavenExecutionException {
        super.afterSessionEnd(session);
    }

    private void configurePIT(MavenSession session) throws MojoFailureException {
        for (MavenProject p : session.getProjects()) {
            Plugin newPlug = new Plugin();
            newPlug.setArtifactId("pitest-maven");
            newPlug.setGroupId("org.pitest");
            newPlug.setVersion("1.5.2");
            // Dependency newDependency = new Dependency();
            // newDependency.setGroupId("org.pitest");
            // newDependency.setArtifactId("pitest-junit5-plugin");
            // newDependency.setVersion("0.8");
            // newPlug.addDependency(newDependency);
            Xpp3Dom configuration = new Xpp3Dom("configuration");
            Xpp3Dom matrix = new Xpp3Dom("fullMutationMatrix");
            matrix.setValue("true");
            Xpp3Dom output = new Xpp3Dom("outputFormats");
            Xpp3Dom outputParam = new Xpp3Dom("param");
            outputParam.setValue("XML");
            configuration.addChild(matrix);
            output.addChild(outputParam);
            configuration.addChild(output);
            newPlug.setConfiguration(configuration);

            p.getBuild().addPlugin(newPlug);
        }
    }
}
