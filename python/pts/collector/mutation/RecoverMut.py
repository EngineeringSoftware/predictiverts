"""
Step1: parse xml report, get {mutator, lineNumber, mutatedClass, sourceFile, descriptions}  --> into json file project_results_dir (project wise)
Step2: Find the file, open the file, find the line, use regex to change
step3: from a list of mut --> all changed files
"""
from typing import *
from seutil import BashUtils, LoggingUtils
from pathlib import Path
import re
import sys

from pts.Environment import Environment

import ipdb

# Mutations = {'PrimitiveReturnsMutator',  # replaced float return with 0.0f for  (Primitive) 0
#              'BooleanTrueReturnValsMutator',  # return true
#              'NullReturnValsMutator',  # return null
#              'BooleanFalseReturnValsMutator',  # return false
#              'ConditionalsBoundaryMutator',  # if < ---> <=
#              'EmptyObjectReturnValsMutator',  # Float to 0,
#              'NegateConditionalsMutator',  # == -> !=
#              'MathMutator',  # + => --
#              'IncrementsMutator',  # ++ -> ==
#              'VoidMethodCallMutator'  # l -
#
#              }


class RecoverMut:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    mutations = {'PrimitiveReturnsMutator',  # replaced float return with 0.0f for  (Primitive) 0
                 'BooleanTrueReturnValsMutator',  # return true
                 'NullReturnValsMutator',  # return null
                 'BooleanFalseReturnValsMutator',  # return false
                 'ConditionalsBoundaryMutator',  # if < ---> <=
                 'EmptyObjectReturnValsMutator',  # Float to 0,
                 'NegateConditionalsMutator',  # == -> !=
                 'MathMutator',  # + => --
                 'IncrementsMutator',  # ++ -> ==
                 'VoidMethodCallMutator',  # l -
                 "InvertNegsMutator",
                 "ReturnValsMutator"
                 }

    CONDITION_MUT_TABLE = {
        "<": "<=",
        "<=": "<",
        ">": ">=",
        ">=": ">"
    }

    NEGATE_CONDITION_MUT_TABLE = {
        "==": "!=",
        "!=": "==",
        "<=": ">",
        ">=": "<",
        "<": ">=",
        ">": "<=",
        "isinstance": "!isinstance"
    }

    PRIMITIVE_RETURN_MUT_TABLE = {
        "int": "0",
        "long": "0",
        "short": "0",
        "float": "0.0f",
        "char": r"\u0000",
        "double": "0.0d",
        "byte": "0",
        "boolean": "true"
    }

    OBJ_RETURN_MUT_TABLE = {
        "Optional.empty": "Optional.empty()",
        "Collections.emptyList": "Collections.emptyList()",
        "Collections.emptySet": "Collections.emptySet()"
    }

    MATH_MUT_TABLE = {
        "+": "-",
        "-": "+",
        "*": "/",
        "/": "*",
        "%": "*",
        "&": "|",
        "|": "&",
        "^": "&",
        "<<": ">>",
        ">>": "<<",
        ">>>": "<<"
    }

    DES_2_OPT = {
        "multiplication": "*",
        "division": "/",
        "subtraction": "-",
        "addition": "+",
        "XOR": "^",
        "AND": "&&",
        "bitwise OR": "|",
        "OR": "||", 
        "modulus": "%",
        "bitwise AND": "&",
        "Shift Left": "<<",
        "Shift Right": ">>",
        "Unsigned Shift Left": "<<<",
        "Unsigned Shift Right": ">>>"
    }

    def __init__(self, output_dir: Path = ""):
        self.output_dir = output_dir
        return

    def recover_code_changes(self, mutator: str, mut_dict: Dict):
        """Recover mutants and return the changed code"""
        source_file = mut_dict["sourceFile"]
        linum = int(mut_dict["lineNumber"])
        # BashUtils.run(f"cp {source_file} {self.output_dir}/", 0)
        new_line, old_line = "", ""
        if mutator not in self.mutations:
            self.logger.warning(f"This mutator {mutator} has not been implemented, the description is " + mut_dict["description"])
        if mutator == "ConditionalsBoundaryMutator":
            new_line, old_line, _ = self.condition_boundary_mut(source_file, linum, mut_dict["description"])
        elif mutator == "NegateConditionalsMutator":
            new_line, old_line, _ = self.negate_condition_mut(source_file, linum, mut_dict["description"])
        elif mutator == "VoidMethodCallMutator":
            new_line, old_line, _ = self.void_method_mut(source_file, linum, mut_dict["description"])
        elif mutator == "NullReturnValsMutator":
            new_line, old_line, _ = self.null_return_mut(source_file, linum)
        elif mutator == "PrimitiveReturnsMutator":
            new_line, old_line, _ = self.primitive_return_mut(source_file, linum, mut_dict["description"])
        elif mutator == "BooleanTrueReturnValsMutator":
            new_line, old_line, _ = self.bool_return_mut(source_file, linum, True)
        elif mutator == "BooleanFalseReturnValsMutator":
            new_line, old_line, _ = self.bool_return_mut(source_file, linum, False)
        elif mutator == "EmptyObjectReturnValsMutator":
            new_line, old_line, _ = self.empty_return_mut(source_file, linum, mut_dict["description"])
        elif mutator == "MathMutator":
            new_line, old_line, _ = self.math_mut(source_file, linum, mut_dict["description"])
        elif mutator == "IncrementsMutator":
            new_line, old_line, _ = self.increment_mut(source_file, linum, mut_dict["description"])
        elif mutator == "InvertNegsMutator":
            new_line, old_line, _ = self.inv_neg_mut(source_file, linum, mut_dict["description"])
        elif mutator == "ReturnValsMutator":
            new_line, old_line, _ = self.ret_val_mut(source_file, linum, mut_dict["description"])
        return new_line, old_line

    def recover_changed_file(self, mutator: str, mut_dict: Dict):
        """Recover mutants and return the changed file (List[lines of code])."""
        source_file = mut_dict["sourceFile"]
        linum = int(mut_dict["lineNumber"])
        # BashUtils.run(f"cp {source_file} {self.output_dir}/", 0)
        list_of_line = []
        new_line, old_line = "", ""
        if mutator not in self.mutations:
            self.logger.warning(
                f"This mutator {mutator} has not been implemented, the description is " + mut_dict["description"])
        if mutator == "ConditionalsBoundaryMutator":
            new_line, old_line, list_of_line = self.condition_boundary_mut(source_file, linum, mut_dict["description"])
        elif mutator == "NegateConditionalsMutator":
            new_line, old_line, list_of_line = self.negate_condition_mut(source_file, linum, mut_dict["description"])
        elif mutator == "VoidMethodCallMutator":
            new_line, old_line, list_of_line = self.void_method_mut(source_file, linum, mut_dict["description"])
        elif mutator == "NullReturnValsMutator":
            new_line, old_line, list_of_line = self.null_return_mut(source_file, linum)
        elif mutator == "PrimitiveReturnsMutator":
            new_line, old_line, list_of_line = self.primitive_return_mut(source_file, linum, mut_dict["description"])
        elif mutator == "BooleanTrueReturnValsMutator":
            new_line, old_line, list_of_line = self.bool_return_mut(source_file, linum, True)
        elif mutator == "BooleanFalseReturnValsMutator":
            new_line, old_line, list_of_line = self.bool_return_mut(source_file, linum, False)
        elif mutator == "EmptyObjectReturnValsMutator":
            new_line, old_line, list_of_line = self.empty_return_mut(source_file, linum, mut_dict["description"])
        elif mutator == "MathMutator":
            new_line, old_line, list_of_line = self.math_mut(source_file, linum, mut_dict["description"])
        elif mutator == "IncrementsMutator":
            new_line, old_line, list_of_line = self.increment_mut(source_file, linum, mut_dict["description"])
        elif mutator == "InvertNegsMutator":
            new_line, old_line, list_of_line = self.inv_neg_mut(source_file, linum, mut_dict["description"])
        elif mutator == "ReturnValsMutator":
            new_line, old_line, list_of_line = self.ret_val_mut(source_file, linum, mut_dict["description"])
        return new_line, old_line, list_of_line

    def inv_neg_mut(self, source_file: Path, linum: int, des: str) -> (str, str, List[str]):
        with open(source_file, 'r', errors="ignore") as f:
            list_of_lines = f.readlines()
        java_line = list_of_lines[linum - 1]
        cleaned_line = re.sub(re.compile("//.*"), "", java_line).rstrip()  # remove inline comments if exists
        if "-" in cleaned_line:
            new_line = re.sub(re.escape("-"), "", cleaned_line)
            list_of_lines[linum - 1] = new_line
            return new_line, cleaned_line, list_of_lines
        else:
            self.logger.info(f"[WARN]: No negation operator found, the line is {cleaned_line}, description is {des}")
            return "", "", []

    def increment_mut(self, source_file: Path, linum: int, des: str) -> (str, str, List[str]):
        with open(source_file, 'r', errors="ignore") as f:
            list_of_lines = f.readlines()
        java_line = list_of_lines[linum - 1]
        cleaned_line = re.sub(re.compile("//.*"), "", java_line).rstrip()  # remove inline comments if exists
        if "++" in cleaned_line:
            new_line = re.sub(re.escape("++"), re.escape("--"), cleaned_line)
            list_of_lines[linum-1] = new_line
            return new_line, cleaned_line, list_of_lines
        if "--" in cleaned_line:
            new_line = re.sub(re.escape("--"), re.escape("++"), cleaned_line)
            list_of_lines[linum - 1] = new_line
            return new_line, cleaned_line, list_of_lines
        if "+=" in cleaned_line:
            new_line = re.sub(re.escape("+="), re.escape("-="), cleaned_line)
            list_of_lines[linum - 1] = new_line
            return new_line, cleaned_line, list_of_lines
        if "-=" in cleaned_line:
            new_line = re.sub(re.escape("-="), re.escape("+="), cleaned_line)
            list_of_lines[linum - 1] = new_line
            return new_line, cleaned_line, list_of_lines
        else:
            try:
                original_code = des.split()[3]
                goal_code = des.split()[5]
                if original_code not in cleaned_line:
                    raise KeyError
                new_line = re.sub(re.escape(original_code), re.escape(goal_code), cleaned_line)
                list_of_lines[linum - 1] = new_line
                return new_line, cleaned_line, list_of_lines
            except:
                self.logger.info(f"[WARN]: No increment operator found, the line is {cleaned_line}, description is {des}, "
                                    f"exception is {sys.exc_info()[0]}.")
                return "", "", []

    def condition_boundary_mut(self, source_file: Path, linum: int, des: str) -> (str, str, List[str]):
        with open(source_file, 'r', errors="ignore") as f:
            list_of_lines = f.readlines()
        java_line = re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).rstrip()  # remove comments
        try:
            # in case multi-line
            while java_line[-1] != ";" and ";" not in java_line:
                linum += 1
                java_line += re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).rstrip()
            # end while
            matched_operator = re.findall("[<>]=?", java_line)
        except:
            return "", "", []
        if len(matched_operator) == 0:
            self.logger.info(
                f"[WARN]: ConditionalsBoundaryMutator exception: no operator matched, please check: {java_line.strip()}.")
            return "", "", []
        else:
            for op in matched_operator:
                changed_op = self.CONDITION_MUT_TABLE[op]
                new_line = re.sub(op, changed_op, java_line)
                list_of_lines[linum - 1] = new_line
            return new_line, java_line, list_of_lines

    def math_mut(self, source_file: Path, linum: int, des: str):
        # We do not consider multi-line math ops
        with open(source_file, 'r', errors="ignore") as f:
            list_of_lines = f.readlines()

        java_line = re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).rstrip()  # remove comments
        try:
            while java_line[-1] != ";" and ";" not in java_line:
                linum += 1
                java_line += re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).rstrip()
                # first delete inline comments
            cleaned_line = java_line
            op_des = str(re.search(r"Replaced (.*) with", des).group(1))
            if "multiplication" in op_des:
                changed_op = self.DES_2_OPT["multiplication"]
            elif "addition" in op_des:
                changed_op = self.DES_2_OPT["addition"]
            elif "division" in op_des:
                changed_op = self.DES_2_OPT["division"]
            elif "subtraction" in op_des:
                changed_op = self.DES_2_OPT["subtraction"]
            elif "modulus" in op_des:
                changed_op = self.DES_2_OPT["modulus"]
            else:
                changed_op = self.DES_2_OPT[op_des]
            new_op = self.DES_2_OPT[str(re.search(r"with (.*)", des).group(1))]
            new_line = re.sub(re.escape(changed_op), re.escape(new_op), cleaned_line)
            list_of_lines[linum - 1] = new_line
            return new_line, java_line, list_of_lines
        except KeyError:
            self.logger.info(
                f"[WARN]: MathMutator: can not find the matched operator, the java line is "
                f"the description is {des}.")
            return "", "", ""
        except:
            self.logger.info(
                     f"[WARN]: MathMutator: unexpected error: {sys.exc_info()[0]}, please ignore.")
            return "", "", ""

        # matched_operator = re.findall("[*+/%&|^-]", cleaned_line)
        # if len(matched_operator) == 0:
        #     if "<<" in cleaned_line:
        #         op = "<<"
        #         changed_op = self.MATH_MUT_TABLE[op]
        #         new_line = re.sub(op, changed_op, cleaned_line)
        #         list_of_lines[linum - 1] = new_line
        #     elif len(re.findall(">>+", cleaned_line)) > 0:
        #         op = re.findall(">>+", cleaned_line)[0]
        #         changed_op = self.MATH_MUT_TABLE[op]
        #         new_line = re.sub(op, changed_op, cleaned_line)
        #         list_of_lines[linum - 1] = new_line
        #     else:
        #         self.logger.warning(f"MathMutator: no match, please check: {java_line}, description is {des}")
        #         return "", ""
        # # end if
        # elif len(matched_operator) == 1:
        #     op = matched_operator[0]
        #     try:
        #         changed_op = self.MATH_MUT_TABLE[op]
        #         new_line = re.sub(re.escape(op), re.escape(changed_op), cleaned_line)
        #         list_of_lines[linum - 1] = new_line
        #     except KeyError:
        #         self.logger.warning(f"MathMutator: operator key error for {op}, please check {cleaned_line.rstrip()}.")
        #     except:
        #         self.logger.warning(
        #             f"MathMutator: unexpected error: {sys.exc_info()[0]}, please check {cleaned_line.rstrip()}.")
        #         return "", ""
        #     # end try
        # # end if
        # elif len(matched_operator) > 1:
        #     try:
        #         changed_op = re.search(r"Replaced (.*) with", des).group(1)
        #         new_op = re.search(r"with (.*)", des).group(1)
        #
        #         des_lst = des.split()
        #         start_idx = des_lst.index("Replaced")
        #         end_idx = des_lst.index("with")
        #         old_op = des_lst[start_idx+1: end_idx]
        #         new_op = des_lst[end_idx+1:]
        #         if len(old_op) == 1 and old_op[0] in self.DES_2_OPT:
        #             changed_op = self.DES_2_OPT[old_op[0]]
        #
        #             new_line = re.sub(re.escape(changed_op), re.escape(self.MATH_MUT_TABLE[changed_op]), cleaned_line)
        #         changed_op = self.DES_2_OPT[des.split()[2]]
        #         if changed_op not in matched_operator:
        #             raise KeyError
        #         new_line = re.sub(re.escape(changed_op), re.escape(self.MATH_MUT_TABLE[changed_op]), cleaned_line)
        #         list_of_lines[linum - 1] = new_line

        # with open(self.output_dir / "changed_code.java", "w") as f:
        #     f.writelines(list_of_lines)

    def negate_condition_mut(self, source_file: Path, linum: int, des: str):
        with open(source_file, 'r', errors="ignore") as f:
            list_of_lines = f.readlines()
        java_line = list_of_lines[linum - 1]
        java_line = re.sub(re.compile("//.*"), "", java_line).rstrip()  # remove inline comments
        matched_operator = re.findall("[<>]=?|[!=]=", java_line)
        # if len(matched_operator) > 1:
        #     self.logger.warning(f"NegateConditionalsMutator: more than one operators are matched, please check: {java_line.rstrip()},"
        #           f"the description is {des}.")
        #     return "", ""
        if len(matched_operator) == 0:
            if "equals" in java_line:
                new_line = []
                for token in java_line.split():
                    if "equals" in token:
                        token = "!" + token
                    # end if
                    new_line.append(token)
                # end for
                list_of_lines[linum - 1] = " ".join(new_line)
                # with open(self.output_dir / "changed_code.java", "w") as f:
                #     f.writelines(list_of_lines)
                return list_of_lines[linum - 1], java_line, list_of_lines
            else:
                return "", "", []
            # end if
            # matched_expression = re.findall("\((.+)\)", java_line)
            # if len(matched_expression) == 0:
            #     self.logger.warning(f"NegateConditionalsMutator: no operator matched, please check: {java_line.rstrip()}, "
            #           f"the description is {des}.")
            #     return "", ""
            # else:
            #     try:
            #         exp = matched_expression[0]
            #         changed_exp = f"!{exp}"
            #         new_line = re.sub(exp, changed_exp, java_line)
            #         list_of_lines[linum - 1] = new_line
            #         # with open(self.output_dir / "changed_code.java", "w") as f:
            #         #     f.writelines(list_of_lines)
            #         return list_of_lines[linum - 1], java_line
            #     except re.error:
            #         self.logger.warning(f"NegateConditionalsMutator: error in matching {java_line.rstrip()}, the description is: {des}.")
            #         return "", ""
        else:
            list_of_lines[linum - 1] = java_line
            for op in matched_operator:
                changed_op = self.NEGATE_CONDITION_MUT_TABLE[op]
                new_line = re.sub(op, changed_op, list_of_lines[linum - 1])
                list_of_lines[linum - 1] = new_line
            # end for

            # with open(self.output_dir / "changed_code.java", "w") as f:
            #     f.writelines(list_of_lines)
            return list_of_lines[linum - 1], java_line, list_of_lines

    def void_method_mut(self, source_file: Path, linum: int, des: str):
        with open(source_file, 'r', errors="ignore") as f:
            list_of_lines = f.readlines()
        java_line = list_of_lines[linum - 1]
        method_call_stmt = re.sub(re.compile("//.*"), "", java_line).rstrip()  # remove inline comments
        start_index = linum - 1
        try:
            while method_call_stmt[-1] != ";" and ";" not in method_call_stmt:
                linum += 1
                method_call_stmt += re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).rstrip()
            # end while
            end_index = linum - 1
        except:
            return "", "", []
        if "::" in des:
            try:
                removed_method_call = des.split("::")[1]
                method_call_stmt_regex = re.compile(f"{removed_method_call}(.*?);")
                new_method_call = method_call_stmt_regex.sub(";", method_call_stmt)
                del list_of_lines[start_index: end_index + 1]
                list_of_lines.insert(start_index, new_method_call)
                return new_method_call, method_call_stmt, list_of_lines
            except:
                self.logger.info(f"[WARN]: Can not do the replacement: {des}")
                return "", "", []
        # print(f"The description can not parse {des}.")
        return "", "", []

    def null_return_mut(self, source_file: Path, linum: int):
        with open(source_file, 'r', errors="ignore") as f:
            list_of_lines = f.readlines()
        java_line = list_of_lines[linum - 1]
        return_stmt = re.sub(re.compile("//.*"), "", java_line).rstrip()
        start_index = linum - 1
        if len(return_stmt) == 0:
            return "", "", []
        while return_stmt[-1] != ";" and ";" not in return_stmt:
            linum += 1
            return_stmt += re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).rstrip()
        end_index = linum - 1
        # delete the relevant statements
        del list_of_lines[start_index: end_index + 1]
        new_return_stmt = "return null;"
        list_of_lines.insert(start_index, new_return_stmt)
        # with open(self.output_dir / "changed_code.java", "w") as f:
        #     f.writelines(list_of_lines)
        return new_return_stmt, return_stmt, list_of_lines

    def primitive_return_mut(self, source_file: Path, linum: int, des: str):
        with open(source_file, 'r', errors="ignore") as f:
            list_of_lines = f.readlines()
        # print(list_of_lines[linum-1])
        return_stmt = re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).rstrip()
        start_index = linum - 1
        if len(return_stmt) == 0:
            return "", "", []
        while return_stmt[-1] != ";" and ";" not in return_stmt:
            linum += 1
            return_stmt += re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).rstrip()
        end_index = linum - 1
        # delete the relevant statements
        del list_of_lines[start_index: end_index + 1]
        if des.split()[0] == "replaced":
            type = des.split()[1]
        else:
            self.logger.info(f"[WARN]: PrimitiveReturnsMutator: descriptions do not obey common rules, please check: {des}")
            return "", "", []
        new_return_stmt = f"return {self.PRIMITIVE_RETURN_MUT_TABLE[type]};"
        list_of_lines.insert(start_index, new_return_stmt)
        # with open(self.output_dir / "changed_code.java", "w") as f:
        #     f.writelines(list_of_lines)
        return new_return_stmt, return_stmt, list_of_lines

    def bool_return_mut(self, source_file: Path, linum: int, true: bool):
        with open(source_file, 'r', errors="ignore") as f:
            list_of_lines = f.readlines()
        return_stmt = re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).rstrip()
        start_index = linum - 1
        if len(return_stmt) == 0:
            return "", "", []
        while return_stmt[-1] != ";" and ";" not in return_stmt:
            linum += 1
            return_stmt += re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).rstrip()
        end_index = linum - 1
        old_stmt = return_stmt
        return_stmt = re.compile(r"return(.*);")
        if true:
            mutated_stmt = "return true;"
            new_return_stmt = return_stmt.sub(mutated_stmt, old_stmt)
            del list_of_lines[start_index: end_index + 1]
            list_of_lines.insert(start_index, new_return_stmt)
        else:
            mutated_stmt = "return false;"
            new_return_stmt = return_stmt.sub(mutated_stmt, old_stmt)
            del list_of_lines[start_index: end_index + 1]
            list_of_lines.insert(start_index, new_return_stmt)
        return new_return_stmt, old_stmt, list_of_lines

    def empty_return_mut(self, source_file: Path, linum: int, des: str):
        with open(source_file, 'r', errors="ignore") as f:
            list_of_lines = f.readlines()
        return_stmt = re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).rstrip()
        start_index = linum - 1
        if len(return_stmt) == 0:
            return "", "", []
        while return_stmt[-1] != ";" and ";" not in return_stmt:
            linum += 1
            return_stmt += re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).rstrip()
        end_index = linum - 1
        # delete the relevant statements
        del list_of_lines[start_index: end_index + 1]
        empty_obj = des.split()[4]
        if empty_obj in self.OBJ_RETURN_MUT_TABLE:
            replaced_obj = self.OBJ_RETURN_MUT_TABLE[empty_obj]
        else:
            replaced_obj = empty_obj
        new_return_stmt = f"return {replaced_obj};"
        list_of_lines.insert(start_index, new_return_stmt)
        # with open(self.output_dir / "changed_code.java", "w") as f:
        #     f.writelines(list_of_lines)
        return new_return_stmt, return_stmt, list_of_lines

    def ret_val_mut(self, source_file: Path, linum: int, des: str):
        with open(source_file, 'r', errors="ignore") as f:
            list_of_lines = f.readlines()
        return_stmt = re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).strip()  # remove inline comment
        start_index = linum - 1
        if len(return_stmt) == 0:
            return "", "", []
        while return_stmt[-1] != ";" and ";" not in return_stmt:
            linum += 1
            return_stmt += re.sub(re.compile("//.*"), "", list_of_lines[linum - 1]).strip()
        end_index = linum - 1
        # delete the relevant statements
        del list_of_lines[start_index: end_index + 1]

        if "null" in des:
            new_return_stmt = "return null;"
        elif "long" in des:
            pat = r"return\s(.*);"
            if re.search(pat, return_stmt):
                value = re.search(pat, return_stmt).group(1)
            else:
                self.logger.info(
                    f"[WARN]: Can not match the return value in statement: {return_stmt}, in file {source_file}, the"
                    f"description is {des}")
                return "", ""
            new_return_stmt = f"return {value}+1;"
        elif "float" in des or "double" in des:
            pat = r"return\s(.*);"
            if re.search(pat, return_stmt):
                value = re.search(pat, return_stmt).group(1)
            else:
                self.logger.info(
                    f"[WARN]: Can not match the return value in statement: {return_stmt}, in file {source_file}, the"
                    f"description is {des}")
                return "", ""
            new_return_stmt = f"return -({value}+1.0);"
        elif "integer" in des:
            new_return_stmt = "return 0;"
        else:
            self.logger.info(f"[WARN]: Can not find the matched mutation, the description is {des}.")
            return "", "", []
        list_of_lines.insert(start_index, new_return_stmt)
        # with open(self.output_dir / "changed_code.java", "w") as f:
        #     f.writelines(list_of_lines)
        return new_return_stmt, return_stmt, list_of_lines
