import os
import os.path as osp
import re
from collections import defaultdict
from enum import Enum
from typing import Any


# General auxiliary functions
def get_last_class_definition_name(filepath: str, line_number: int) -> str:
    """Get the name of the last class definition in (filepath) before (line_number)"""
    with open(filepath, "r") as file:
        lines = file.readlines()
    for line in lines[:line_number]:
        matches = re.findall(r"class (.*?)[\(:]", line)
        if matches:
            return matches[0]
    return ""


def get_associated_function_name(filepath: str, line_number: int) -> str:
    """Given a (filepath) and a (line_number), return the name of the function that is associated with the line number:
    if the line number points to a decorator, return the name of the function that is decorated.
    Otherwise, return the name of the function encompasses the line number."""
    # Retrieve file's content
    with open(filepath, "r") as file:
        lines = file.readlines()
    # Check the given line number: if it's associated with a decorator, we go down, otherwise we go up
    if re.match(r"\s*@", lines[line_number - 1]):
        lines = lines[line_number - 1 :]
    else:
        lines = lines[:line_number]
        lines.reverse()
    # Iterate over the lines to find the first function definition
    for line in lines:
        matches = re.findall(r"\s*def (.*?)\(", line)
        if matches:
            return matches[0]
    return ""


def group_by_common_suffix(d: dict[str, list[Any]], suffix: str) -> dict[str, list[Any]]:
    """Group all lists that end with the suffix into a single list, and add it to the dict with a new key."""
    # Aggregate all lists that end with the suffix
    keys_to_pop, grouped = [], []
    for key, lst in d.items():
        if key.endswith(suffix):
            keys_to_pop.append(key)
            grouped.extend(lst)
    # If any key had the suffix, add the aggregated list to the dict
    if keys_to_pop:
        d["... " + suffix] = grouped
    # Remove all keys that were aggregated
    for key in keys_to_pop:
        d.pop(key)
    return d


# Parsing-related auxiliary functions
def parse_passed(line: str) -> str:
    """Parse a passed test report line, eg: PASSED file.py::TestClass::test_name"""
    line = line.removeprefix("PASSED ")
    chunks = line.split("::")
    if len(chunks) == 3:
        file, parent_class, name = chunks
    else:
        file, parent_class, name = chunks[0], "", chunks[-1]
    return TestReport(TestStatus.PASSED, file, name, "", parent_class, 1)


def parse_failed(line: str) -> str:
    """Parse a failed test report line, eg: FAILED file.py::TestClass::test_name - msg"""
    line = line.removeprefix("FAILED ")
    test_provenance, msg = line.split(" - ", maxsplit=1)
    chunks = test_provenance.split("::")
    if len(chunks) == 3:
        file, parent_class, name = chunks
    else:
        file, parent_class, name = chunks[0], "", chunks[-1]
    return TestReport(TestStatus.FAILED, file, name, msg, parent_class, 1)


def parse_skipped(line: str, tests_root: str) -> str:
    """Parse a skipped test report line, eg: SKIPPED [n_skipped] file:line_nb: msg"""
    og_line = line + ""
    line = line.removeprefix("SKIPPED ")
    multiplicity, line = line.split("] ", maxsplit=1)
    multiplicity = int(multiplicity[1:])
    file, line = line.split(":", 1)
    line_nb_str, msg = line.split(": ", maxsplit=1)
    try:
        name = get_associated_function_name(osp.join(tests_root, file), int(line_nb_str))
        parent_class = get_last_class_definition_name(osp.join(tests_root, file), int(line_nb_str))
    except Exception as e:
        print(f"Error getting first test name for {file}:{line_nb_str}: {e} . Original line: {og_line}")
        name = "UNKNOWN NAME"
        parent_class = ""
    return TestReport(TestStatus.SKIPPED, file, name, msg, parent_class, multiplicity)


# TODO: Implement this
def parse_error(line: str) -> str:
    """Parse an error test report line, eg: ERROR file.py::TestClass::test_name - msg"""
    line = line.removeprefix("ERROR ")
    return TestReport(TestStatus.ERROR, "", "", "", "", 1)


# Test-related classes
class TestStatus(Enum):
    """Enum for the status of a test."""

    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


class TestReport:
    """Class to represent the result of a single test."""

    def __init__(
        self, status: TestStatus, file: str, name: str, msg: str, parent_class: str = "", multiplicity: int = 1
    ):
        self.status = status
        self.file = file
        self.name = name
        self.msg = msg
        self.parent_class = parent_class
        self.multiplicity = multiplicity

    def __repr__(self):
        if self.parent_class:
            return f"{self.status.value} {self.file}::{self.parent_class}::{self.name} - {self.msg}"
        return f"{self.status.value} {self.file}::{self.name} - {self.msg}"

    @classmethod
    def from_line(cls, line: str, tests_root: str) -> "TestReport":
        line = line.strip()
        if line.startswith("PASSED "):
            return parse_passed(line)
        elif line.startswith("FAILED "):
            return parse_failed(line)
        elif line.startswith("SKIPPED "):
            return parse_skipped(line, tests_root)
        elif line.startswith("ERROR "):
            return parse_error(line)
        else:
            return TestReport(TestStatus.UNKNOWN, "", "", "", "", 1)


class PytestReport:
    """Class to represent the result of a pytest run."""

    def __init__(self, pytest_dir: str, tests_root: str = "tests", no_parsing: bool = False) -> None:
        assert osp.isdir(pytest_dir), f"Reports dir {pytest_dir} does not exist"
        self.pytest_dir = pytest_dir
        self.tests_root = tests_root
        self.tests = [] if no_parsing else self.parse_summary()

    def deepcopy(self) -> "PytestReport":
        report = PytestReport(self.pytest_dir, self.tests_root, no_parsing=True)
        report.tests = self.tests[:]
        return report

    def parse_summary(self) -> list[TestReport]:
        with open(osp.join(self.pytest_dir, "summary_short.txt")) as file:
            lines = file.readlines()
        parsed = []
        for line in lines:
            try:
                parsed.append(TestReport.from_line(line, self.tests_root))
            except BaseException as e:
                print(line, e)
        return [test for test in parsed if test.status != TestStatus.UNKNOWN]

    def filter_by_status(self, status: TestStatus) -> list[TestReport]:
        return [test for test in self.tests if test.status == status]

    def search_equivalent_test(
        self, test: TestReport, remove_if_found: bool = False, check_parent_class: bool = False
    ) -> TestReport:
        """Search for test with the same name and file in the report."""
        for t in self.tests:
            equivalent = (t.name == test.name) and (t.file == test.file)
            if check_parent_class:
                equivalent = equivalent and (t.parent_class == test.parent_class)
            if equivalent:
                if remove_if_found:
                    self.tests.remove(t)
                return t
        return TestReport(TestStatus.UNKNOWN, "", "", "__Equivalent test not found__")

    def compare_to(
        self, other: "PytestReport", check_parent_class: bool = False, verbose: bool = False
    ) -> dict[tuple[TestStatus, TestStatus], list[tuple[TestReport, TestReport]]]:
        """Compare the tests in the report to another report."""
        # Display the number of tests in both reports
        if verbose:
            print(f"Self has {len(self.tests)} tests while other has {len(other.tests)}")
        # For each test in the report, assign it to a category
        categories = defaultdict(lambda: [])
        for test in self.tests:
            equivalent_test = other.search_equivalent_test(
                test, remove_if_found=True, check_parent_class=check_parent_class
            )
            categories[(test.status, equivalent_test.status)].append((test, equivalent_test))
        # Add the remaining tests in other
        categories[(TestStatus.UNKNOWN, TestStatus.UNKNOWN)].extend((other.tests, other.tests))
        # Display the number of tests in each category
        if verbose:
            for (status, equivalent_status), tests in categories.items():
                print(f"{status} -> {equivalent_status}: {len(tests)}")
        return categories

    def count_statuses(self, with_multiplicity: bool = False) -> dict[TestStatus, int]:
        status_to_count = defaultdict(int)
        for test in self.tests:
            status_to_count[test.status] += test.multiplicity if with_multiplicity else 1
        return status_to_count


class BatchedPytestReports:
    """Class to represent a directory containing multiple pytest reports."""

    def __init__(self, main_dir: str, name: str = "", tests_root: str = ".") -> None:
        """
        Args:
            main_dir: The directory containing the pytest reports.
            name: The name of the batch.
            tests_root: The path to the directory containing the test scripts.
        """
        self.main_dir = main_dir
        self.name = name
        self.tests_root = tests_root
        # Get all the pytest report directories
        pytest_dirs = sorted([f for f in os.listdir(self.main_dir) if osp.isdir(osp.join(self.main_dir, f))])
        self.pytest_reports = {
            pytest_dir: PytestReport(osp.join(self.main_dir, pytest_dir), self.tests_root)
            for pytest_dir in pytest_dirs
        }

    def find_common_reports(
        self, other: "BatchedPytestReports", verbose: bool = False
    ) -> tuple[list[str], list[str], list[str]]:
        """Find the common and exclusive pytest reports between two batches."""
        common = sorted(set(self.pytest_reports.keys()).intersection(set(other.pytest_reports.keys())))
        exclusive_to_self = sorted(set(self.pytest_reports.keys()) - set(other.pytest_reports.keys()))
        exclusive_to_other = sorted(set(other.pytest_reports.keys()) - set(self.pytest_reports.keys()))
        if verbose:
            print(f"Found {len(exclusive_to_self)} models exclusive to {self.name}")
            print(f"Found {len(exclusive_to_other)} models exclusive to {other.name}")
            print(f"Found {len(common)} models in common")
        return common, exclusive_to_self, exclusive_to_other

    def get_all_skips(self, group_by_common_suffixes: bool = True) -> dict[str, list[str]]:
        """Get all the skips in the reports, optionally grouped by common suffixes."""
        all_skips = {}
        for _, batched_report in self.pytest_reports.items():
            for test in batched_report.tests:
                if test.status == TestStatus.SKIPPED:
                    key, msg = test.file + "::" + test.name, test.msg
                    if msg not in all_skips:
                        all_skips[msg] = []
                    all_skips[msg].append(key)
        if group_by_common_suffixes:
            all_skips = group_by_common_suffix(all_skips, "does not support SDPA")
            all_skips = group_by_common_suffix(all_skips, "is not compatible with torch.fx")
            all_skips = group_by_common_suffix(all_skips, "is not a priorited model for now")
        return all_skips

    def compare_one_pytest_report(
        self, pytest_name: str, other: "BatchedPytestReports", check_parent_class: bool = False, verbose: bool = False
    ) -> dict[tuple[TestStatus, TestStatus], list[tuple[TestReport, TestReport]]]:
        """Deep dive on a pytest report, comparing it to another report."""
        # Assert that the pytest report exists in both report and reference
        assert pytest_name in self.pytest_reports, f"Report {pytest_name} not found in report {self.name}"
        assert pytest_name in other.pytest_reports, f"Report {pytest_name} not found in reference{other.name}"
        # Display the number of tests in the pytest run
        self_pytest_report = self.pytest_reports[pytest_name].deepcopy()
        other_pytest_report = other.pytest_reports[pytest_name].deepcopy()
        return self_pytest_report.compare_to(other_pytest_report, check_parent_class, verbose=verbose)

    def filter_by_status_change(
        self, other: "BatchedPytestReports", change_pattern: str, check_parent_class: bool = False
    ) -> dict[str, list[tuple[TestReport, TestReport]]]:
        """Filter the tests in the report by the status change."""
        # Infer target key
        self_pattern, other_pattern = change_pattern.split(" -> ")
        # Retrieve the names common reports
        common_reports = self.find_common_reports(other, verbose=False)[0]
        # Filter the tests in the common reports
        filtered_tests = defaultdict(list)
        for report_name in common_reports:
            compared = self.compare_one_pytest_report(report_name, other, check_parent_class, verbose=False)
            for key, tests in compared.items():
                self_pattern_match = re.match(self_pattern, key[0].value) is not None
                other_pattern_match = re.match(other_pattern, key[1].value) is not None
                if self_pattern_match and other_pattern_match:
                    filtered_tests[report_name].extend(tests)
        return filtered_tests

    def filter_by_msg(self, msg_part: str) -> list[TestReport]:
        filtered_tests = []
        for _, tests in self.pytest_reports.items():
            for test in tests.filter_by_status(TestStatus.SKIPPED):
                if msg_part in test.msg:
                    filtered_tests.append(test)
        return filtered_tests

    def rerun_skipped_tests(self, msg_part: str, report_name: str) -> str:
        tests_names = set()
        for report_name, tests in self.pytest_reports.items():
            for test in tests.filter_by_status(TestStatus.SKIPPED):
                if msg_part in test.msg:
                    tests_names.add(test.name)
        tests_names = sorted(tests_names)
        cmd = f"RUN_SLOW=1 pytest tests/models -rsfE -v --make-reports={report_name} -k '"
        cmd += " or ".join(tests_names)
        cmd += "'"
        return cmd

    def count_statuses(self, verbose: bool = False, with_multiplicity: bool = False) -> dict[TestStatus, int]:
        status_to_count = defaultdict(int)
        for _, tests in self.pytest_reports.items():
            for test in tests.tests:
                status_to_count[test.status] += test.multiplicity if with_multiplicity else 1
        if verbose:
            for status, count in sorted(status_to_count.items(), key=lambda x: x[1], reverse=True):
                print(f"{status}: {count}")
            total = sum(status_to_count.values())
            print(f"Total: {total}")
        return status_to_count


if __name__ == "__main__":
    import argparse

    # Parse the path to two reports
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathA", type=str, required=True)
    parser.add_argument("--pathB", type=str, required=True)
    parser.add_argument("--comparison-mode", type=str, default="=", choices=["=", ">"])
    args = parser.parse_args()

    # Parse reports
    reportsA = BatchedPytestReports(args.pathA, tests_root="/home/remi/transformers/")
    reportsB = BatchedPytestReports(args.pathB, tests_root="/home/remi/transformers/")

    # Print the models that are not common to both reports
    common, exclusive_to_A, exclusive_to_B = reportsA.find_common_reports(reportsB)
    print(f"\n{exclusive_to_A = }\n{exclusive_to_B = }\n")

    # For each model, if the status count is not the same, print the difference
    for model_name in common:
        status_count_A = reportsA.pytest_reports[model_name].count_statuses(with_multiplicity=True)
        status_count_B = reportsB.pytest_reports[model_name].count_statuses(with_multiplicity=True)

        # Choose to display the model depending on comparison mode
        if args.comparison_mode == "=":
            everything_ok = status_count_A == status_count_B
        elif args.comparison_mode == ">":
            everything_ok = all([
                status_count_B[TestStatus.PASSED] >= status_count_A[TestStatus.PASSED] and
                status_count_B[TestStatus.FAILED] <= status_count_A[TestStatus.FAILED] and
                status_count_B[TestStatus.ERROR] == 0
            ])
        else:
            raise ValueError(f"Invalid comparison mode: {args.comparison_mode}")

        if not everything_ok:
            status_count_A_str = ", ".join([f"{status.name}: {count}" for status, count in status_count_A.items()])
            status_count_B_str = ", ".join([f"{status.name}: {count}" for status, count in status_count_B.items()])
            print(f"{model_name}:\n{status_count_A_str}\n{status_count_B_str}\n")
