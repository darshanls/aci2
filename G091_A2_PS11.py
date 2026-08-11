"""
Bayesian Network for Predictive Maintenance - PS11
Group: G091
BITS Pilani WILP - Artificial and Computational Intelligence Assignment 2

This module implements a Bayesian Network for industrial predictive maintenance.
It supports two scenarios:
  1. Bearing Failure Detection (variables: B, T, V)
  2. Cooling System Failure Detection (variables: C, S1, S2)

The implementation uses a reusable inference engine based on joint enumeration
for exact probabilistic inference.
"""

import sys
import itertools


# ---------------------------------------------------------------------------
# Input Parsing Module
# ---------------------------------------------------------------------------

def parse_input(file_path):
    """
    Parse the input file and extract scenario parameters.

    Expected format:
        PSXX
        SCENARIO_1_BEARING_FAILURE
        P_B=<value>
        P_T_given_B=<value>
        ...
        SCENARIO_2_COOLING_SYSTEM
        P_C=<value>
        ...

    Returns:
        dict with keys 'scenario1' and 'scenario2', each containing
        a dict of probability parameters.

    Raises:
        ValueError: on malformed lines, missing keys, duplicate keys,
                    or invalid probability values.
    """
    scenarios = {}
    current_scenario = None
    seen_scenarios = set()

    valid_scenario_ids = {
        "SCENARIO_1_BEARING_FAILURE": "scenario1",
        "SCENARIO_2_COOLING_SYSTEM": "scenario2",
    }

    required_keys = {
        "scenario1": [
            "P_B", "P_T_given_B", "P_T_given_notB",
            "P_V_given_B", "P_V_given_notB",
        ],
        "scenario2": [
            "P_C", "P_S1_given_C", "P_S1_given_notC",
            "P_S2_given_C", "P_S2_given_notC",
        ],
    }

    try:
        with open(file_path, "r") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading input file: {e}")

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        # Skip the header line (e.g., PS11)
        if line.startswith("PS") and line[2:].isdigit():
            continue

        # Check for scenario identifier
        if line in valid_scenario_ids:
            scenario_key = valid_scenario_ids[line]
            if scenario_key in seen_scenarios:
                raise ValueError(
                    f"Line {line_num}: Duplicate scenario identifier '{line}'."
                )
            seen_scenarios.add(scenario_key)
            current_scenario = scenario_key
            scenarios[current_scenario] = {}
            continue

        # Check for invalid scenario identifiers
        if line.startswith("SCENARIO_") and line not in valid_scenario_ids:
            raise ValueError(
                f"Line {line_num}: Invalid scenario identifier '{line}'."
            )

        # Parse key=value pairs
        if "=" not in line:
            raise ValueError(
                f"Line {line_num}: Malformed input line '{line}'. "
                "Expected key=value format."
            )

        if current_scenario is None:
            raise ValueError(
                f"Line {line_num}: Parameter '{line}' found before any "
                "scenario identifier."
            )

        parts = line.split("=", 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(
                f"Line {line_num}: Malformed input line '{line}'."
            )

        key = parts[0].strip()
        val_str = parts[1].strip()

        # Check for duplicate keys within a scenario
        if key in scenarios[current_scenario]:
            raise ValueError(
                f"Line {line_num}: Duplicate key '{key}' in "
                f"{current_scenario}."
            )

        # Validate probability value
        try:
            value = float(val_str)
        except ValueError:
            raise ValueError(
                f"Line {line_num}: Invalid probability value '{val_str}' "
                f"for key '{key}'. Must be a numeric value."
            )

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"Line {line_num}: Probability value {value} for key "
                f"'{key}' is out of range [0, 1]."
            )

        scenarios[current_scenario][key] = value

    # Validate that all required scenarios are present
    for sc_key in ["scenario1", "scenario2"]:
        if sc_key not in scenarios:
            raise ValueError(
                f"Missing scenario: {sc_key}. Both scenarios are required."
            )
        for req_key in required_keys[sc_key]:
            if req_key not in scenarios[sc_key]:
                raise ValueError(
                    f"Missing required key '{req_key}' in {sc_key}."
                )

    return scenarios


# ---------------------------------------------------------------------------
# Bayesian Network Core
# ---------------------------------------------------------------------------

class BayesianNetwork:
    """
    A generic Bayesian Network for three binary variables with the structure:
        Root -> Child1
        Root -> Child2

    The network supports:
      - Joint probability table construction
      - Exact inference via joint enumeration
      - Marginal independence testing
      - Conditional independence testing given the root variable
    """

    def __init__(self, var_names, prior_root, cpt_child1, cpt_child2):
        """
        Initialise the Bayesian Network.

        Args:
            var_names: tuple of (root_name, child1_name, child2_name)
            prior_root: P(Root = True)
            cpt_child1: dict with keys True/False ->
                         P(Child1=True | Root=key)
            cpt_child2: dict with keys True/False ->
                         P(Child2=True | Root=key)
        """
        self.root_name, self.child1_name, self.child2_name = var_names
        self.var_names = var_names
        self.prior_root = prior_root
        self.cpt_child1 = cpt_child1
        self.cpt_child2 = cpt_child2

        # Build the joint probability table once
        self.joint_table = self._build_joint_table()

    def _p_root(self, root_val):
        """Return P(Root = root_val)."""
        return self.prior_root if root_val else (1.0 - self.prior_root)

    def _p_child1_given_root(self, child1_val, root_val):
        """Return P(Child1 = child1_val | Root = root_val)."""
        p = self.cpt_child1[root_val]
        return p if child1_val else (1.0 - p)

    def _p_child2_given_root(self, child2_val, root_val):
        """Return P(Child2 = child2_val | Root = root_val)."""
        p = self.cpt_child2[root_val]
        return p if child2_val else (1.0 - p)

    def _joint_prob(self, root_val, child1_val, child2_val):
        """
        Compute joint probability using the factorization:
            P(Root, Child1, Child2) = P(Root) * P(Child1|Root) * P(Child2|Root)
        """
        return (
            self._p_root(root_val)
            * self._p_child1_given_root(child1_val, root_val)
            * self._p_child2_given_root(child2_val, root_val)
        )

    def _build_joint_table(self):
        """
        Construct the complete joint probability table.

        Returns:
            list of tuples: [(root_val, child1_val, child2_val, prob), ...]
            ordered with True values first.
        """
        table = []
        for root_val, child1_val, child2_val in itertools.product(
            [True, False], repeat=3
        ):
            prob = self._joint_prob(root_val, child1_val, child2_val)
            table.append((root_val, child1_val, child2_val, prob))
        return table

    def query(self, query_var, evidence):
        """
        Perform exact inference using joint enumeration.

        Computes P(query_var = True | evidence) by summing over all
        consistent rows in the joint table.

        Args:
            query_var: str - name of the query variable
            evidence: dict mapping variable names to True/False values

        Returns:
            float - posterior probability P(query_var = True | evidence)

        Raises:
            ValueError: if query_var is not in the network or if
                        evidence contains unknown variables.
        """
        var_index = {
            self.root_name: 0,
            self.child1_name: 1,
            self.child2_name: 2,
        }

        if query_var not in var_index:
            raise ValueError(f"Unknown query variable: {query_var}")
        for ev_var in evidence:
            if ev_var not in var_index:
                raise ValueError(f"Unknown evidence variable: {ev_var}")

        q_idx = var_index[query_var]

        numerator = 0.0
        denominator = 0.0

        for row in self.joint_table:
            vals = row[:3]
            prob = row[3]

            # Check if this row is consistent with evidence
            consistent = True
            for ev_var, ev_val in evidence.items():
                if vals[var_index[ev_var]] != ev_val:
                    consistent = False
                    break

            if consistent:
                denominator += prob
                if vals[q_idx] is True:
                    numerator += prob

        if denominator == 0.0:
            raise ValueError(
                "Evidence has zero probability; cannot condition on "
                "impossible evidence."
            )

        return numerator / denominator

    def marginal(self, var_name):
        """
        Compute the marginal probability P(var_name = True).

        Args:
            var_name: str - variable name

        Returns:
            float - P(var_name = True)
        """
        var_index = {
            self.root_name: 0,
            self.child1_name: 1,
            self.child2_name: 2,
        }
        if var_name not in var_index:
            raise ValueError(f"Unknown variable: {var_name}")

        idx = var_index[var_name]
        total = 0.0
        for row in self.joint_table:
            if row[idx] is True:
                total += row[3]
        return total

    def joint_marginal(self, var1_name, var2_name):
        """
        Compute P(var1 = True, var2 = True) by marginalising over
        the third variable.

        Args:
            var1_name, var2_name: str - variable names

        Returns:
            float - P(var1=True, var2=True)
        """
        var_index = {
            self.root_name: 0,
            self.child1_name: 1,
            self.child2_name: 2,
        }
        idx1 = var_index[var1_name]
        idx2 = var_index[var2_name]

        total = 0.0
        for row in self.joint_table:
            if row[idx1] is True and row[idx2] is True:
                total += row[3]
        return total

    def check_marginal_independence(self, var1_name, var2_name):
        """
        Test whether two variables are marginally independent:
            P(var1, var2) == P(var1) * P(var2)

        Args:
            var1_name, var2_name: str - variable names

        Returns:
            bool - True if marginally independent, False otherwise
        """
        p_var1 = self.marginal(var1_name)
        p_var2 = self.marginal(var2_name)
        p_joint = self.joint_marginal(var1_name, var2_name)

        # Use a tolerance for floating point comparison
        return abs(p_joint - p_var1 * p_var2) < 1e-9

    def check_conditional_independence(self, var1_name, var2_name,
                                       given_name):
        """
        Test whether var1 and var2 are conditionally independent given
        the conditioning variable, for both values (True and False).

        Checks: P(var1, var2 | given) == P(var1 | given) * P(var2 | given)
        for given = True and given = False.

        Args:
            var1_name, var2_name: str - the two variables to test
            given_name: str - the conditioning variable

        Returns:
            bool - True if conditionally independent given the variable
        """
        var_index = {
            self.root_name: 0,
            self.child1_name: 1,
            self.child2_name: 2,
        }

        idx1 = var_index[var1_name]
        idx2 = var_index[var2_name]
        idx_given = var_index[given_name]

        for given_val in [True, False]:
            # Sum P(all | given=given_val)
            p_given = 0.0
            p_v1_given = 0.0
            p_v2_given = 0.0
            p_v1_v2_given = 0.0

            for row in self.joint_table:
                vals = row[:3]
                prob = row[3]

                if vals[idx_given] == given_val:
                    p_given += prob
                    if vals[idx1] is True:
                        p_v1_given += prob
                    if vals[idx2] is True:
                        p_v2_given += prob
                    if vals[idx1] is True and vals[idx2] is True:
                        p_v1_v2_given += prob

            if p_given == 0.0:
                continue

            # Conditional probabilities
            p_v1_cond = p_v1_given / p_given
            p_v2_cond = p_v2_given / p_given
            p_joint_cond = p_v1_v2_given / p_given

            if abs(p_joint_cond - p_v1_cond * p_v2_cond) > 1e-9:
                return False

        return True

    def get_joint_table_formatted(self):
        """
        Return the joint probability table as a formatted string.

        Returns:
            str - formatted table with headers and rows
        """
        lines = []
        hdr = (
            f"{self.root_name}   {self.child1_name}   "
            f"{self.child2_name}   Probability"
        )
        lines.append(hdr)

        for row in self.joint_table:
            root_str = "T" if row[0] else "F"
            child1_str = "T" if row[1] else "F"
            child2_str = "T" if row[2] else "F"
            prob_str = f"{row[3]:.4f}"
            lines.append(
                f"{root_str}   {child1_str}   {child2_str}   {prob_str}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scenario Processing
# ---------------------------------------------------------------------------

def process_scenario1(params):
    """
    Process Scenario 1: Bearing Failure Detection.

    Network: B -> T, B -> V
    Factorization: P(B, T, V) = P(B) * P(T|B) * P(V|B)

    Args:
        params: dict of probability parameters

    Returns:
        list of str - output lines
    """
    bn = BayesianNetwork(
        var_names=("B", "T", "V"),
        prior_root=params["P_B"],
        cpt_child1={
            True: params["P_T_given_B"],
            False: params["P_T_given_notB"],
        },
        cpt_child2={
            True: params["P_V_given_B"],
            False: params["P_V_given_notB"],
        },
    )

    output = []

    # Joint probability table
    output.append("Scenario 1 - Joint probability table")
    output.append(bn.get_joint_table_formatted())
    output.append("")

    # Posterior probabilities
    output.append("Scenario 1: Bearing Failure Detection")
    output.append("")

    p_b_given_t = bn.query("B", {"T": True})
    output.append(f"P(B | T) = {p_b_given_t:.4f}")

    p_b_given_v = bn.query("B", {"V": True})
    output.append(f"P(B | V) = {p_b_given_v:.4f}")

    p_b_given_t_v = bn.query("B", {"T": True, "V": True})
    output.append(f"P(B | T and V) = {p_b_given_t_v:.4f}")

    output.append("")

    # Marginal independence test
    marg_indep = bn.check_marginal_independence("T", "V")
    if marg_indep:
        output.append(
            "Temperature and Vibration are independent without evidence."
        )
    else:
        output.append(
            "Temperature and Vibration are not independent "
            "without evidence."
        )

    # Conditional independence test
    cond_indep = bn.check_conditional_independence("T", "V", "B")
    if cond_indep:
        output.append(
            "Temperature and Vibration are conditionally independent "
            "given Bearing Failure."
        )
    else:
        output.append(
            "Temperature and Vibration are not conditionally independent "
            "given Bearing Failure."
        )

    return output


def process_scenario2(params):
    """
    Process Scenario 2: Cooling System Failure.

    Network: C -> S1, C -> S2
    Factorization: P(C, S1, S2) = P(C) * P(S1|C) * P(S2|C)

    Args:
        params: dict of probability parameters

    Returns:
        list of str - output lines
    """
    bn = BayesianNetwork(
        var_names=("C", "S1", "S2"),
        prior_root=params["P_C"],
        cpt_child1={
            True: params["P_S1_given_C"],
            False: params["P_S1_given_notC"],
        },
        cpt_child2={
            True: params["P_S2_given_C"],
            False: params["P_S2_given_notC"],
        },
    )

    output = []

    output.append("Scenario 2: Cooling System Failure")
    output.append("")

    p_c_given_s1 = bn.query("C", {"S1": True})
    output.append(f"P(C | S1) = {p_c_given_s1:.4f}")

    p_c_given_s2 = bn.query("C", {"S2": True})
    output.append(f"P(C | S2) = {p_c_given_s2:.4f}")

    p_c_given_s1_s2 = bn.query("C", {"S1": True, "S2": True})
    output.append(f"P(C | S1 and S2) = {p_c_given_s1_s2:.4f}")

    output.append("")

    # Marginal independence test
    marg_indep = bn.check_marginal_independence("S1", "S2")
    if marg_indep:
        output.append(
            "Sensor Alarm 1 and Sensor Alarm 2 are independent "
            "without evidence."
        )
    else:
        output.append(
            "Sensor Alarm 1 and Sensor Alarm 2 are not independent "
            "without evidence."
        )

    # Conditional independence test
    cond_indep = bn.check_conditional_independence("S1", "S2", "C")
    if cond_indep:
        output.append(
            "Sensor Alarm 1 and Sensor Alarm 2 are conditionally "
            "independent given Cooling Failure."
        )
    else:
        output.append(
            "Sensor Alarm 1 and Sensor Alarm 2 are not conditionally "
            "independent given Cooling Failure."
        )

    return output


# ---------------------------------------------------------------------------
# Output Writer
# ---------------------------------------------------------------------------

def write_output(output_lines, file_path):
    """
    Write the output lines to the specified file.

    Args:
        output_lines: list of str - lines to write
        file_path: str - path to the output file
    """
    try:
        with open(file_path, "w") as fh:
            fh.write("\n".join(output_lines) + "\n")
    except IOError as e:
        raise IOError(f"Error writing output file: {e}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    """
    Main function that orchestrates the Bayesian Network inference pipeline.

    Reads from inputPS11.txt, processes both scenarios, and writes results
    to outputPS11.txt.
    """
    input_file = "inputPS11.txt"
    output_file = "outputPS11.txt"

    # Parse command-line arguments if provided
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]

    try:
        scenarios = parse_input(input_file)
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    output_lines = []

    # Process Scenario 1
    try:
        scenario1_output = process_scenario1(scenarios["scenario1"])
        output_lines.extend(scenario1_output)
    except (ValueError, KeyError) as e:
        print(f"Error processing Scenario 1: {e}")
        sys.exit(1)

    output_lines.append("")

    # Process Scenario 2
    try:
        scenario2_output = process_scenario2(scenarios["scenario2"])
        output_lines.extend(scenario2_output)
    except (ValueError, KeyError) as e:
        print(f"Error processing Scenario 2: {e}")
        sys.exit(1)

    # Write output
    try:
        write_output(output_lines, output_file)
        print(f"Output written to {output_file}")
    except IOError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
