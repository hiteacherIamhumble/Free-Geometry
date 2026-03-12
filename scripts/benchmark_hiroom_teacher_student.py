#!/usr/bin/env python3
"""
Backward-compatible entrypoint.

The multi-dataset implementation now lives in:
  scripts/benchmark_teacher_student_all_datasets.py
"""

from benchmark_teacher_student_all_datasets import main


if __name__ == "__main__":
    main()
