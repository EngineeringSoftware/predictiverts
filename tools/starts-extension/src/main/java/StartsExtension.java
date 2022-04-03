import org.apache.maven.AbstractMavenLifecycleParticipant;
import org.apache.maven.MavenExecutionException;
import org.apache.maven.execution.MavenSession;
import org.apache.maven.model.PluginExecution;
import org.codehaus.plexus.component.annotations.Component;
import org.apache.maven.project.MavenProject;
import org.apache.maven.model.Plugin;
import org.apache.maven.plugin.MojoFailureException;
import org.codehaus.plexus.util.xml.Xpp3Dom;

@Component( role = AbstractMavenLifecycleParticipant.class, hint = "starts")
public class StartsExtension extends AbstractMavenLifecycleParticipant{

    @Override
    public void afterProjectsRead(MavenSession session) throws MavenExecutionException {
        try {
            configureSTARTS(session);
        } catch (MojoFailureException mfe) {
            throw new MavenExecutionException("Unable to install Starts", mfe);
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

    private void configureSTARTS(MavenSession session) throws MojoFailureException {
        for (MavenProject p : session.getProjects()) {
//            Plugin surefirePlug = new Plugin();
//            surefirePlug.setArtifactId("maven-surefire-plugin");
//            surefirePlug.setGroupId("org.apache.maven.plugins");
//            surefirePlug.setVersion("3.0.0-M5");
//            p.getBuild().addPlugin(surefirePlug);
            if (p.getProperties().containsKey("test")){
                System.out.println("find test and remove it");
                p.getProperties().remove("test");
            }

            // Find surefire plugin or its configuration
            p.getBuild().getPlugins().stream()
                    .filter(StartsExtension::isSurefirePlugin)
                    .forEach(StartsExtension::modifySurefirePlugin);
            p.getBuild().getPluginManagement().getPlugins().stream()
                    .filter(StartsExtension::isSurefirePlugin)
                    .forEach(StartsExtension::modifySurefirePlugin);

            Plugin newPlug = new Plugin();
            newPlug.setArtifactId("starts-maven-plugin");
            newPlug.setGroupId("edu.illinois");
            newPlug.setVersion("1.5-SNAPSHOT");
            p.getBuild().addPlugin(newPlug);
        }
    }

    private static boolean isSurefirePlugin(Plugin plugin) {
        return plugin.getGroupId().equals("org.apache.maven.plugins") && plugin.getArtifactId().equals("maven-surefire-plugin");
    }

    private static void modifySurefirePlugin(Plugin plugin) {
        plugin.setVersion("2.22.2");
    }
}
