import xml.etree.ElementTree as ET
from pathlib import Path
from typing import *

from seutil import IOUtils, BashUtils


"""
mutation_dict:
{
   'detected': 'false',
   'status': 'NO_COVERAGE',
   'numberOfTestsRun': '0',
   'sourceFile': 'RDF4J.java', 
   'mutatedClass': 'org.apache.commons.rdf.rdf4j.RDF4J', 
   'mutatedMethod': 'asDataset', 
   'methodDescription': '', 
   'lineNumber': '363', 
   'mutator': 'NullReturnValsMutator', 
   'index': '22', 
   'block': '4', 
   'killingTests': "All" / list(), 
   'succeedingTests': None, / "All", "Remaining", 
   'description': 'replaced return value with null for org/apache/commons/rdf/rdf4j/RDF4J::asDataset'}
"""

class XMLParser:

    @staticmethod
    def parse_pit_report(pom_file: Path, project_name: str, output_dir: Path, default=True):
        """Parse xml pit report and output to a json file, ignore mutations whose status is not "KILLED" or
        "SURVIVED". """
        if default:
            output_file = output_dir / f"{project_name}-default-mutation-report.json"
        else:
            output_file = output_dir / f"{project_name}-all-mutation-report.json"
        tree = ET.parse(pom_file)
        root = tree.getroot()
        mutation_report_list: List[dict] = list()
        for child in root:  # each mutation node
            mutation_dict = child.attrib
            if mutation_dict["status"] == "KILLED":
                for gchild in child:
                    if gchild.tag == "mutator":
                        mutator = gchild.text.split(".")[-1]
                        mutation_dict["mutator"] = mutator
                    elif gchild.tag  == "killingTests" and gchild.text is not None:
                        test_list = gchild.text.split("|")
                        mutation_dict[str(gchild.tag)] = []
                        for t in test_list:
                            if '(' in t:
                                mutation_dict[str(gchild.tag)].append((t[t.find("(") + 1:t.find(")")], t[:t.find("(")]))
                            else:
                                mutation_dict[str(gchild.tag)].append((t, t))
                    elif gchild.tag == "succeedingTests" and gchild.text is not None:
                        test_list = gchild.text.split("|")
                        mutation_dict[str(gchild.tag)] = []
                        for t in test_list:
                            if '(' in t:
                                mutation_dict[str(gchild.tag)].append((t[t.find("(") + 1:t.find(")")], t[:t.find("(")]))
                            else:
                                mutation_dict[str(gchild.tag)].append((t, t))
                    else:
                        mutation_dict[str(gchild.tag)] = gchild.text
                    # end if
                # end for
                mutation_report_list.append(mutation_dict)
                # end for
            elif mutation_dict["status"] == "TIMED_OUT":
                for gchild in child:
                    if gchild.tag == "mutator":
                        mutator = gchild.text.split(".")[-1]
                        mutation_dict["mutator"] = mutator
                    elif gchild.tag == "killingTests":
                        mutation_dict[str(gchild.tag)] = "All"
                    else:
                        mutation_dict[str(gchild.tag)] = gchild.text
                    # end if
                # end for
                mutation_report_list.append(mutation_dict)
            elif mutation_dict["status"] in ["SURVIVED", "NO_COVERAGE"]:
                for gchild in child:
                    if gchild.tag == "mutator":
                        mutator = gchild.text.split(".")[-1]
                        mutation_dict["mutator"] = mutator
                    elif gchild.tag == "succeedingTests" and gchild.text is not None:
                        test_list = gchild.text.split("|")
                        mutation_dict[str(gchild.tag)] = []
                        for t in test_list:
                            if '(' in t:
                                mutation_dict[str(gchild.tag)].append((t[t.find("(") + 1:t.find(")")], t[:t.find("(")]))
                            else:
                                mutation_dict[str(gchild.tag)].append((t, t))
                    else:
                        mutation_dict[str(gchild.tag)] = gchild.text
                    # end if
                # end for
                mutation_report_list.append(mutation_dict)
            # end if
        # end for
        IOUtils.dump(output_file, mutation_report_list)

    @staticmethod
    def parse_pit_report_labels(pom_file: Path, project_name: str, output_dir: Path, default=True):
        """Parse xml pit report and output to a json file, ignore mutations whose status is not "KILLED" or
        "SURVIVED". """
        if default:
            output_file = output_dir / f"{project_name}-default-mutation-report.json"
        else:
            output_file = output_dir / f"{project_name}-all-mutation-report.json"
        tree = ET.parse(pom_file)
        root = tree.getroot()
        mutation_report_list: List[dict] = list()
        for child in root:  # each mutation node
            mutation_dict = child.attrib
            if mutation_dict["status"] == "KILLED":
                for gchild in child:
                    if gchild.tag == "mutator":
                        mutator = gchild.text.split(".")[-1]
                        mutation_dict["mutator"] = mutator
                    elif gchild.tag == "killingTests" and gchild.text is not None:
                        test_list = gchild.text.split("|")
                        mutation_dict[str(gchild.tag)] = []
                        for t in test_list:
                            if '(' in t:
                                mutation_dict[str(gchild.tag)].append((t[t.find("(") + 1:t.find(")")], t[:t.find("(")]))
                            else:
                                mutation_dict[str(gchild.tag)].append((t, t))
                    elif gchild.tag == "succeedingTests":
                        try:
                            test_list = gchild.text.split("|")
                            mutation_dict[str(gchild.tag)] = [(t[t.find("(") + 1:t.find(")")], t[:t.find("(")])
                                                              for t in test_list]  # [(test_class, test_method), ...]
                        except AttributeError:
                            mutation_dict[str(gchild.tag)] = gchild.text
                    else:
                        mutation_dict[str(gchild.tag)] = gchild.text
                    # end if
                mutation_dict["nocoverageTests"] = "Remaining"
                # end for
                mutation_report_list.append(mutation_dict)
                # end for
            elif mutation_dict["status"] == "TIMED_OUT":
                for gchild in child:
                    if gchild.tag == "mutator":
                        mutator = gchild.text.split(".")[-1]
                        mutation_dict["mutator"] = mutator
                    elif gchild.tag == "killingTests":
                        mutation_dict[str(gchild.tag)] = "All"
                    else:
                        mutation_dict[str(gchild.tag)] = gchild.text
                    # end if
                # end for
                mutation_report_list.append(mutation_dict)
            elif mutation_dict["status"] in ["SURVIVED"]:
                for gchild in child:
                    if gchild.tag == "mutator":
                        mutator = gchild.text.split(".")[-1]
                        mutation_dict["mutator"] = mutator
                    elif gchild.tag == "succeedingTests":
                        test_list = gchild.text.split("|")
                        mutation_dict[str(gchild.tag)] = [(t[t.find("(") + 1:t.find(")")], t[:t.find("(")])
                                                          for t in test_list]  # [(test_class, test_method), ...]
                    else:
                        mutation_dict[str(gchild.tag)] = gchild.text
                    # end if
                mutation_dict["nocoverageTests"] = "Remaining"
                # end for
                mutation_report_list.append(mutation_dict)
            elif mutation_dict["status"] == "NO_COVERAGE":
                for gchild in child:
                    if gchild.tag == "mutator":
                        mutator = gchild.text.split(".")[-1]
                        mutation_dict["mutator"] = mutator
                    mutation_dict[str(gchild.tag)] = gchild.text
                mutation_dict["nocoverageTests"] = "All"
            # end if
        # end for
        IOUtils.dump(output_file, mutation_report_list)
