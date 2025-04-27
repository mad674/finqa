#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone solver for FINQA-style programs over numeric and table operations.
Usage:
  python solve_program.py "<program>" [<table_csv_file>]

If <table_csv_file> is provided, it should be a CSV with:
  row_name, val1, val2, ...
otherwise only pure numeric operations are supported.
"""
import sys, csv, re
from sympy import simplify

# --- Utility to parse numbers ---
def str_to_num(text):
    t = text.replace("$", "").replace(",", "").strip()
    if t.lower().startswith("const_"):
        t = t[6:]
    if t.endswith("%"):
        t = t[:-1]
        return float(t) / 100.0
    return float(t)

# --- Table row processing ---
def process_row(cells):
    nums = []
    for cell in cells:
        try:
            nums.append(str_to_num(cell))
        except:
            # skip or error
            raise ValueError(f"Cannot parse numeric cell '{cell}'")
    return nums

# --- DSL Parsing ---
def parse_steps_from_program(program: str):
    program = program.strip()
    parts, depth, start = [], 0, 0
    for i, c in enumerate(program):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            parts.append(program[start:i].strip())
            start = i + 1
    parts.append(program[start:].strip())
    steps = [p for p in parts if p]
    steps.append("<EOF>")
    return steps

# split a step into tokens
def tokenize_dsl_line(line: str):
    return re.findall(r"[a-zA-Z_]+|\#\d+|[(),]|[-+]?\d*\.?\d+", line)

# --- Program evaluator ---
ops = {"add","subtract","multiply","divide","exp","greater",
       "table_sum","table_average","table_max","table_min"}

def eval_program(tokens, table=None):
    # drop trailing EOF
    if tokens and tokens[-1] == "EOF":
        tokens = tokens[:-1]
    # must be 6 tokens per step: op,(,a1,,,a2,)
    if len(tokens) % 6 != 0:
        raise ValueError("Invalid token length")
    memory = {}
    for step_idx in range(len(tokens)//6):
        base = step_idx*6
        op,_,a1,_,a2,_ = tokens[base:base+6]
        if op not in ops:
            raise ValueError(f"Unsupported op '{op}'")
        # resolve arg1
        if a1.startswith('#'):
            v1 = memory[int(a1[1:])]
        else:
            v1 = str_to_num(a1)
        # resolve arg2
        if a2.startswith('#'):
            v2 = memory[int(a2[1:])]
        else:
            v2 = str_to_num(a2)

        # dispatch
        if op == "add": res = v1 + v2
        elif op == "subtract": res = v1 - v2
        elif op == "multiply": res = v1 * v2
        elif op == "divide": res = v1 / v2
        elif op == "exp": res = v1 ** v2
        elif op == "greater": res = "yes" if v1 > v2 else "no"
        else:
            # table ops require table
            if table is None:
                raise ValueError("Table operations need a table CSV.")
            # build row map
            rowmap = {row[0]: row[1:] for row in table}
            # for table ops arg1 must be row name or memory
            if a1.startswith('#'):
                idx = int(a1[1:])
                arr = memory[idx]
            else:
                arr = process_row(rowmap[a1])
            if op == "table_sum": res = sum(arr)
            elif op == "table_average": res = sum(arr)/len(arr)
            elif op == "table_max": res = max(arr)
            elif op == "table_min": res = min(arr)
            else:
                raise ValueError(f"Unknown table op '{op}'")
        memory[step_idx] = res

    final = memory[len(memory)-1]
    if isinstance(final, float):
        final = round(final,5)
    return final


def evaluate_program(prog, table_csv=None):
    result = None
    steps = parse_steps_from_program(prog)
    toks = []
    for s in steps:
        if s == '<EOF>':
            toks.append('EOF')
        else:
            toks.extend(tokenize_dsl_line(s))
    try:
        result = eval_program(toks, table=table_csv)
        print("result:",result)
    except Exception as e:
        print("Error:", e)
        result = "Error evaluating program"
    
    return result
