# rr = BashUtils.run("mvn -Dexec.executable='bash' -Dexec.args='-c '\"'\"'echo ${project.groupId}:${project.artifactId} ${project.packaging} ${PWD}'\"'\"'' exec:exec -q")
import getpass
import requests
import shutil
import subprocess
import sys
import fnmatch
import os
import pathlib
from pts.Macros import Macros
from seutil import BashUtils
import json
import re

def find_file(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def find_directory(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

# From list of slugs, filter for Maven projects
def filter_for_maven_projects(slugs):
    maven_projects = []
    for project in slugs:
        request = 'https://github.com/' + project + '/blob/master/pom.xml'  # Assume branch of master only...
        response = requests.get(request)
        # If can get a response, then project is (probably) Maven
        if response.ok:
            # Some tweaking to get the actual project slug, in case of redirects
            actual_project_slug = response.url.replace('https://github.com/', '').replace('/blob/master/pom.xml', '')
            maven_projects.append(actual_project_slug)
    return maven_projects

# From list of slugs, filter for Gradle projects
def filter_for_gradle_projects(slugs):
    gradle_projects = []
    for project in slugs:
        request = 'https://github.com/' + project + '/blob/master/build.gradle' # Assume branch of master only...
        response = requests.get(request)
        # If can get a response, then project is (probably) Maven
        if response.ok:
            # Some tweaking to get the actual project slug, in case of redirects
            actual_project_slug = response.url.replace('https://github.com/', '').replace('/blob/master/build.gradle', '')
            gradle_projects.append(actual_project_slug)
    return gradle_projects

# From list of valid Maven projects, filter for ones that are on Travis
def filter_for_travis_projects(maven_projects):
    travis_projects = []
    for project in maven_projects:
        # First check if on GitHub they have a .travis.yml
        request = 'https://github.com/' + project + '/blob/master/.travis.yml'  # Assume branch of master only...
        response = requests.get(request)
        # If cannot get a response, then project is not Travis, so can skip
        if not response.ok:
            continue

        # Otherwise, hit the Travis API to double-check it has been activated
        request = 'https://api.travis-ci.org/repos/' + project
        response = requests.get(request)
        if response.ok:
            try:
                data = json.loads(response.text, encoding = 'utf-8')
            except ValueError:
                # Something went wrong, Travis returns some weird image of sorts, so skip
                print('TRAVIS FILTER VALUE ERROR FOR ' + project)
                continue
            if data['active']:
                travis_projects.append(project)
    return travis_projects

# From list of valid Travis projects, filter for ones that are multimodule
def filter_for_multimodule_projects(travis_projects):
    multimodule_projects = []
    for project in travis_projects:
        command = 'git clone https://github.com/' + project + ' tmp --depth=1'
        subprocess.call(command.split())
        if len(find_file('pom.xml', 'tmp')) > 1:
        #if len(find_file('build.gradle', 'tmp')) > 1:
            multimodule_projects.append(project)
        shutil.rmtree('tmp')
    return multimodule_projects

# From list of valid Travis projects, filter for ones that are multimodule with tests
def filter_for_multimodule_with_test_projects(multimodule_projects):
    multimodule_projects_with_tests = []
    for project in multimodule_projects:
        # checke the travis_ci log
        project = project.strip()
        r = re.compile(r"@ (?P<name>\S*) ---.*?T E S T S", flags=re.DOTALL)
        project_info = BashUtils.run(f'curl https://api.travis-ci.org/repos/{project}').stdout
        if project_info:
            try:
                data = json.loads(project_info, encoding='utf-8')
                repo_id = data['id']
                last_build_id = data['last_build_number']
                response = requests.get(f'https://api.travis-ci.org/builds?after_number={last_build_id}&repository_id={repo_id}',
                              headers={"Content-Type": "application/json", "Accept":"application/vnd.travis-ci.2+json"})
                if response.ok:
                    builds = json.loads(response.text, encoding = 'utf-8')
                    module_num = 0
                    module_related_job_id = 0

                    for build in builds['builds']:
                        job_ids = build['job_ids']
                        for job_id in job_ids:
                            log_info = BashUtils.run(f"curl https://api.travis-ci.org/v3/job/{job_id}/log.txt").stdout
                            log_info = log_info.replace("\r", "")
                            multi_module_projects_set = {m.group("name") for m in r.finditer(log_info)}
                            if module_num < len(multi_module_projects_set):
                                module_num = len(multi_module_projects_set)
                                module_related_job_id = job_id
                            # if module_num < log_info.count(pattern):
                            #     module_num = log_info.count(pattern)
                            #     module_related_job_id = job_id
                    multimodule_projects_with_tests.append((project, module_num, module_related_job_id))
                    print(project, module_num, module_related_job_id)
            except ValueError:
                continue

    return multimodule_projects_with_tests


def travis_project_helper(uname):
    passwd = getpass.getpass()
    travis_file = pathlib.Path(Macros.results_dir / 'travis_projects.txt')
    if not travis_file.is_file():
        slugs = []
        url = 'https://api.github.com/search/repositories?q=language:java&sort=stars&order=desc&per_page=100'
        for i in range(1, 11):
            suffix = '&page=' + str(i)
            request = url + suffix
            response = requests.get(request, auth=(uname, passwd))
            if response.ok:
                data = json.loads(response.text)
                for k in data['items']:
                    slugs.append(k['full_name'])
            else:
                break

        print('ALL PROJECTS:', len(slugs))

        # Check if the project is a Maven project by merely checking if a link to the pom.xml can be accessed
        maven_projects = filter_for_maven_projects(slugs)
        print('MAVEN PROJECTS:', len(maven_projects))

        # Check if the Maven projects are on Travis, by hitting the Travis API and checking that it's active
        travis_projects = filter_for_travis_projects(maven_projects)
        print('TRAVIS PROJECTS:', len(travis_projects))

        with open(travis_file, 'a') as out_file:
            for travis_project in travis_projects:
                out_file.write(travis_project)
                out_file.write("\n")


def main(uname, out_file):
    # uname = args[1] # Username
    # out_file = args[2]
    passwd = getpass.getpass()
    # If multi_module.txt exists, then just make use of it
    multimodule_file = pathlib.Path(Macros.results_dir / 'multi_module.txt')
    if multimodule_file.is_file():
        with open(multimodule_file) as f:
            multimodule_list = f.readlines()
            multimodule_projects_with_tests = filter_for_multimodule_with_test_projects(multimodule_list)
            with open(out_file, 'w') as out:
                for project, num, job_id in multimodule_projects_with_tests:
                    out.write(project + " " + str(num) + " " + str(job_id) + '\n')
    else:
        # Get all the Java projects on GitHub
        slugs = []
        url = 'https://api.github.com/search/repositories?q=language:java&sort=stars&order=desc&per_page=100'
        for i in range(1, 11):
            suffix = '&page=' + str(i)
            request = url + suffix
            response = requests.get(request, auth=(uname, passwd))
            if response.ok:
                data = json.loads(response.text)
                for k in data['items']:
                    slugs.append(k['full_name'])
            else:
                break

        print('ALL PROJECTS:', len(slugs))

        # Check if the project is a Maven project by merely checking if a link to the pom.xml can be accessed
        maven_projects = filter_for_maven_projects(slugs)
        print('MAVEN PROJECTS:', len(maven_projects))

        # Check if the Maven projects are on Travis, by hitting the Travis API and checking that it's active
        travis_projects = filter_for_travis_projects(maven_projects)
        print('TRAVIS PROJECTS:', len(travis_projects))

        # For each such project, check if it's multi-module by checking it out and counting that it has more than one pom.xml
        multimodule_projects = filter_for_multimodule_projects(travis_projects)
        print('MULTIMODULE PROJECTS', len(multimodule_projects))

        # For each such project, check if it's multi-module with test by checking its maven log on travis ci.
        multimodule_with_test_projects = filter_for_multimodule_with_test_projects(multimodule_projects)
        print('MULTIMODULE PROJECTS with test', len(multimodule_with_test_projects))

        # Print out final filtered list of projects (the slugs)
        with open(out_file, 'w') as out:
            for project in multimodule_projects:
                out.write(project + '\n')

if __name__ == '__main__':
    main(sys.argv)