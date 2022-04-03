from typing import *
import re


class SubTokenizer:

    MULTI_CHAR_OPS = ["++", "--", "<<", ">>", ">>>", "<=", ">=", "==", "!=", "&&", "||",
        "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=", "<<=", ">>=", ">>>=",
        "->", "//", "/*", "/**", "*/", "::"]
    MULTI_CHAR_OPS.sort(key=lambda x: len(x), reverse=True)  # Ensure longer ops are before shorter ops

    RE_SPLITTER_WS = re.compile(r"\s+")
    RE_SPLITTER_WORD = re.compile(r"([a-zA-Z0-9_]+)")
    RE_DECIMAL = re.compile(r"\d*\.\d*f?")

    @classmethod
    def is_java_identifier(cls, s: str) -> bool:
        return all([c.isalnum() or c == "_" for c in s])

    @classmethod
    def sub_tokenize_java_like(cls, s: str) -> List[str]:
        """
        Sub-tokenizes the string to sub-tokens using Java-like way (applies to both Java code and Java comments which may contain identifiers).

        Some auxiliary sub-tokens are inserted to help recover the original tokens:
        - SOST and EOST are inserted to contain each sub-tokens sequence that belong to the same token; they're not inserted for a token that doesn't need a sub-tokenization.
        - UP is inserted before each sub-token that was UPPER case (including a single UPPER case letter), and the sub-token will be converted to small case.
        - Cp is inserted before each sub-token that was Capital case, and the sub-token will be converted to small case.

        TODO: Java internal names (which may appear in method ids) are not well handled, e.g., "Ljava" will become a word.
        :param s: the string to sub-tokenize
        :return: the sub-tokens
        """
        sub_tokens = list()

        # First split on whitespace
        tokens = cls.RE_SPLITTER_WS.split(s)

        bracket_stack = list()  # Only used for distinguishing ">>" and "> >", ">>>" and "> > >". Ignore the matching errors if any

        while len(tokens) > 0:
            t_by_ws = tokens.pop(0)
            if t_by_ws is None or t_by_ws == "":  continue

            # Check if it is special token (begins with "$", ends with "$") or a double literal (number with decimal)
            if (len(t_by_ws) > 2 and t_by_ws[0] == "$" and t_by_ws[-1] == "$") or (cls.RE_DECIMAL.fullmatch(t_by_ws)):
                sub_tokens.append(t_by_ws)
                continue
            else:
                # If not, split on word boundaries
                almost_sub_tokens = cls.RE_SPLITTER_WORD.split(t_by_ws)
                while len(almost_sub_tokens) > 0:
                    t = almost_sub_tokens.pop(0)
                    if t is None or t == "":  continue
                    if not cls.is_java_identifier(t):
                        # If not alnum, take the longest possible operators and put the rest to next sub-token
                        if len(t) > 1:
                            found_longer_op = False
                            for op in cls.MULTI_CHAR_OPS:
                                if t.startswith(op):
                                    almost_sub_tokens.insert(0, t[len(op):])
                                    t = op
                                    found_longer_op = True
                                    break
                                # end if
                            # end for
                            if not found_longer_op:
                                almost_sub_tokens.insert(0, t[1:])
                                t = t[0]
                            # end for
                        # end if

                        # Maintain bracket stack and check if this step should prefer ">"
                        if t == "(":  bracket_stack.append("(")
                        elif t == "[":  bracket_stack.append("[")
                        elif t == "{":  bracket_stack.append("{")
                        elif t == "<":  bracket_stack.append("<")
                        elif t == ")":
                            # Pop all "<"
                            while len(bracket_stack) > 0 and bracket_stack[-1] == "<":  bracket_stack.pop()
                            if len(bracket_stack) > 0 and bracket_stack[-1] == "(":  bracket_stack.pop()
                        elif t == "]":
                            # Pop all "<"
                            while len(bracket_stack) > 0 and bracket_stack[-1] == "<":  bracket_stack.pop()
                            if len(bracket_stack) > 0 and bracket_stack[-1] == "[":  bracket_stack.pop()
                        elif t == "}":
                            # Pop all "<"
                            while len(bracket_stack) > 0 and bracket_stack[-1] == "<":  bracket_stack.pop()
                            if len(bracket_stack) > 0 and bracket_stack[-1] == "{":  bracket_stack.pop()
                        elif t == ">":
                            if len(bracket_stack) > 0 and bracket_stack[-1] == "<":  bracket_stack.pop()
                        elif t == ">>":
                            if len(bracket_stack) > 1 and bracket_stack[-1] == bracket_stack[-2] == "<":
                                # Prefer > >
                                t = ">"
                                almost_sub_tokens.insert(0, ">")
                                bracket_stack.pop()
                            # end if
                        elif t == ">>>":
                            if len(bracket_stack) > 0 and bracket_stack[-1] == "<":
                                # Prefer > ?
                                t = ">"
                                almost_sub_tokens.insert(0, ">>")
                                bracket_stack.pop()
                            # end if
                        # end if

                        sub_tokens.append(t)
                    else:
                        # Detect CamelCase and UPPER_CASE
                        token_sub_tokens = list()
                        cur_sub_token = ""
                        last_is_upper = False
                        last_is_number = False
                        for c in t:
                            if c == "_":
                                if len(cur_sub_token) > 0:  token_sub_tokens.append(cur_sub_token)
                                cur_sub_token = ""
                                token_sub_tokens.append(c)
                                last_is_number = False
                                last_is_upper = False
                            elif c.isdigit():
                                if not last_is_number:
                                    # Finish current sub-token
                                    if len(cur_sub_token) > 0:  token_sub_tokens.append(cur_sub_token)
                                    cur_sub_token = ""
                                # end if
                                cur_sub_token += c
                                last_is_number = True
                                last_is_upper = False
                            elif c.isupper():
                                if not last_is_upper:
                                    # Finish current sub_token
                                    if len(cur_sub_token) > 0:  token_sub_tokens.append(cur_sub_token)
                                    cur_sub_token = ""
                                # end if
                                cur_sub_token += c
                                last_is_upper = True
                                last_is_number = False
                            elif c.islower():
                                if last_is_number:
                                    # Finish current sub-token
                                    if len(cur_sub_token) > 0: token_sub_tokens.append(cur_sub_token)
                                    cur_sub_token = ""
                                # end if
                                if last_is_upper:
                                    # Finish previous full UPPER case word, but keep current Capital case word
                                    if len(cur_sub_token) > 1: token_sub_tokens.append(cur_sub_token[:-1])
                                    cur_sub_token = cur_sub_token[-1:]
                                # end if
                                cur_sub_token += c
                                last_is_number = False
                                last_is_upper = False
                            # end if
                        # end for
                        if len(cur_sub_token) > 0:  token_sub_tokens.append(cur_sub_token)

                        # if len(token_sub_tokens) > 1:  sub_tokens.append(MLConsts.vocab_bost)
                        for st in token_sub_tokens:
                            # if st.isupper():
                            #     sub_tokens.append(MLConsts.vocab_up)
                            # elif st[0].isupper():
                            #     sub_tokens.append(MLConsts.vocab_cp)
                            # end if
                            sub_tokens.append(st.lower())
                        # end for
                        # if len(token_sub_tokens) > 1:  sub_tokens.append(MLConsts.vocab_eost)
                    # end if
                # end for
            # end if
        # end for

        return sub_tokens