package edu.illinois.extension;

import java.util.Collections;
import java.util.Properties;

import org.apache.maven.AbstractMavenLifecycleParticipant;
import org.apache.maven.MavenExecutionException;
import org.apache.maven.execution.MavenSession;
import org.codehaus.plexus.component.annotations.Component;
import org.apache.maven.project.MavenProject;
import org.apache.maven.model.Plugin;
import org.codehaus.plexus.util.xml.Xpp3Dom;
import org.apache.maven.model.PluginExecution;
import org.apache.maven.plugin.MojoFailureException;

// your extension must be a "Plexus" component so mark it with the annotation
@Component( role = AbstractMavenLifecycleParticipant.class, hint = "ekstazi")
public class EkstaziExtension extends AbstractMavenLifecycleParticipant
{

    @Override
    public void afterSessionStart( MavenSession session )
        throws MavenExecutionException
    {
        // System.out.println("OWOLABI(AfterSessionStart): ");
    }

    @Override
    public void afterSessionEnd( MavenSession session )
        throws MavenExecutionException
    {
        // System.out.println("OWOLABI(AfterSessionEnd): ");
    }

    @Override
    public void afterProjectsRead( MavenSession session )
        throws MavenExecutionException
    {
        boolean found = false;
        for (MavenProject project : session.getProjects()) {
            if (project.getProperties().containsKey("test")){
                System.out.println("find test and remove it");
                project.getProperties().remove("test");
            }
            for (Plugin plugin : project.getBuildPlugins()) {
                if (plugin.getArtifactId().equals("maven-surefire-plugin") && plugin.getGroupId().equals("org.apache.maven.plugins")) {
                    found = true;
                    System.out.println("Modifying surefire to set surefire version...");
                    System.out.println("=====Version(before):: " + plugin.getVersion());
                    plugin.setVersion("2.22.2");
                    System.out.println("=====Version(after):: " + plugin.getVersion());
                }
            }
        }

        if (!found) {
            System.out.println("=====I couldn't find surefire plugin!!!!");
        }

        System.out.println("Adding Ekstazi to the list of plugins...");
        try {
            configureEkstazi(session);
        } catch (MojoFailureException mfe) {
            throw new MavenExecutionException("Unable to install Ekstazi", mfe);
        }
    }


    private void configureEkstazi(MavenSession session) throws MojoFailureException {
       for (MavenProject p : session.getProjects()) {
           Plugin newPlug = new Plugin();
           newPlug.setArtifactId("ekstazi-maven-plugin");
           newPlug.setGroupId("org.ekstazi");
           newPlug.setVersion("5.3.0");

           PluginExecution ex = new PluginExecution();
           ex.setId("ekstazi");
           ex.setGoals(Collections.singletonList("select"));
           ex.setPhase("process-test-classes");
           newPlug.addExecution(ex);
           p.getBuild().addPlugin(newPlug);
       }
    }

}
